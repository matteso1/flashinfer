import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from packaging.version import Version

import flashinfer
from flashinfer import aot
from flashinfer.jit import core, cpp_ext
from flashinfer.jit.attention import modules as attention_modules
from flashinfer.utils import (
    PosEncodingMode,
    determine_attention_backend,
    is_fa3_prefill_head_dim_supported,
)
from tests.test_helpers import jit_utils


def test_nvcc_parallelism_flags_use_flashinfer_nvcc_threads(monkeypatch):
    monkeypatch.setenv("FLASHINFER_NVCC_THREADS", "4")

    assert cpp_ext.get_nvcc_parallelism_flags() == ["--threads=4"]


def test_nvcc_parallelism_flags_ignore_sccache_launcher(monkeypatch):
    monkeypatch.setenv("FLASHINFER_NVCC_THREADS", "4")
    monkeypatch.setenv("FLASHINFER_NVCC_LAUNCHER", "sccache")

    assert cpp_ext.get_nvcc_parallelism_flags() == ["--threads=4"]


def test_generate_ninja_uses_sccache_compatible_nvcc_depfile_flag(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(cpp_ext, "get_cuda_path", lambda: "/usr/local/cuda")
    monkeypatch.setattr(cpp_ext.jit_env, "FLASHINFER_JIT_DIR", tmp_path / "jit")
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "7.5")

    ninja = cpp_ext.generate_ninja_build_for_op(
        name="test_module",
        sources=[tmp_path / "generated" / "kernel.cu"],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_dirs=None,
    )

    assert "--generate-dependencies-with-compile -MF $out.d" in ninja
    assert "--dependency-output" not in ninja


def test_debug_jit_uses_sccache_compatible_nvcc_device_debug_flag(monkeypatch):
    monkeypatch.setenv("FLASHINFER_JIT_DEBUG", "1")
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)
    monkeypatch.setattr(core, "get_nvcc_parallelism_flags", lambda: ["--threads=1"])

    spec = core.gen_jit_spec(
        name="test_module",
        sources=[],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_paths=None,
    )

    assert "--device-debug" in spec.extra_cuda_cflags
    assert "-G" not in spec.extra_cuda_cflags


def test_release_jit_propagates_ndebug_to_host_cflags(monkeypatch):
    monkeypatch.delenv("FLASHINFER_JIT_DEBUG", raising=False)
    monkeypatch.delenv("FLASHINFER_JIT_VERBOSE", raising=False)
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)
    monkeypatch.setattr(core, "get_nvcc_parallelism_flags", lambda: ["--threads=1"])

    spec = core.gen_jit_spec(
        name="test_module",
        sources=[],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_paths=None,
    )

    assert "-DNDEBUG" in spec.extra_cflags
    assert "-DNDEBUG" in spec.extra_cuda_cflags


def test_debug_jit_does_not_propagate_ndebug(monkeypatch):
    monkeypatch.setenv("FLASHINFER_JIT_DEBUG", "1")
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)
    monkeypatch.setattr(core, "get_nvcc_parallelism_flags", lambda: ["--threads=1"])

    spec = core.gen_jit_spec(
        name="test_module",
        sources=[],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_paths=None,
    )

    assert "-DNDEBUG" not in spec.extra_cflags
    assert "-DNDEBUG" not in spec.extra_cuda_cflags


def test_run_ninja_uses_max_jobs(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setenv("MAX_JOBS", "8")
    monkeypatch.setattr(cpp_ext.subprocess, "run", fake_run)

    cpp_ext.run_ninja(tmp_path, tmp_path / "build.ninja", verbose=False)

    assert commands == [
        [
            "ninja",
            "-v",
            "-C",
            str(tmp_path.resolve()),
            "-f",
            str((tmp_path / "build.ninja").resolve()),
            "-j",
            "8",
        ]
    ]


def test_jit_spec_build_rewrites_ninja_before_build(monkeypatch):
    writes = []
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)
    monkeypatch.setattr(cpp_ext, "get_cuda_path", lambda: "/usr/local/cuda")

    spec = core.JitSpecNvcc(
        name="test_module",
        sources=[],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_dirs=None,
    )

    monkeypatch.setattr(spec, "write_ninja", lambda: writes.append(True))
    monkeypatch.setattr(core, "run_ninja", lambda *_args, **_kwargs: None)

    spec.build(verbose=False, need_lock=False)

    assert writes == [True]


def _test_jit_spec(name: str, source: Path, extra_cuda_cflags=None):
    return core.JitSpecNvcc(
        name=name,
        sources=[source],
        extra_cflags=None,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=None,
        extra_include_dirs=None,
    )


def test_jit_build_identity_changes_with_compile_inputs(monkeypatch, tmp_path):
    monkeypatch.setattr(cpp_ext, "get_cuda_path", lambda: "/usr/local/cuda")
    monkeypatch.setattr(core.jit_env, "FLASHINFER_JIT_DIR", tmp_path / "jit")
    source = tmp_path / "sources" / "kernel.cu"

    first = _test_jit_spec("test_module", source, ["-DFIRST"])
    second = _test_jit_spec("test_module", source, ["-DSECOND"])

    assert first.build_dir != second.build_dir
    assert first.lock_path != second.lock_path


def test_jit_build_identity_tracks_cuda_home_in_same_process(monkeypatch, tmp_path):
    nvcc_calls = []

    def fake_check_output(command, **_kwargs):
        nvcc_calls.append(command[0])
        return "Cuda compilation tools, release 13.0, V13.0.0"

    monkeypatch.setattr(cpp_ext.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(core.jit_env, "FLASHINFER_JIT_DIR", tmp_path / "jit")
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "8.9")
    source = tmp_path / "sources" / "kernel.cu"
    spec = _test_jit_spec("test_module", source)

    first_cuda_home = tmp_path / "cuda-first"
    second_cuda_home = tmp_path / "cuda-second"
    monkeypatch.setenv("CUDA_HOME", str(first_cuda_home))
    first_build_dir = spec.build_dir
    monkeypatch.setenv("CUDA_HOME", str(second_cuda_home))
    second_build_dir = spec.build_dir

    assert first_build_dir != second_build_dir
    assert nvcc_calls == [
        str(first_cuda_home / "bin" / "nvcc"),
        str(second_cuda_home / "bin" / "nvcc"),
    ]


def test_jit_ninja_is_relocatable_and_scoped_to_its_build_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(cpp_ext, "get_cuda_path", lambda: "/usr/local/cuda")
    source = tmp_path / "sources" / "kernel.cu"
    roots = [tmp_path / "first-root", tmp_path / "second-root"]
    generated = []
    relative_build_dirs = []

    for root in roots:
        monkeypatch.setattr(core.jit_env, "FLASHINFER_JIT_DIR", root)
        spec = _test_jit_spec("test_module", source)
        spec.write_ninja()
        generated.append(spec.ninja_path.read_text())
        relative_build_dirs.append(spec.build_dir.relative_to(root))

    assert relative_build_dirs[0] == relative_build_dirs[1]
    assert generated[0] == generated[1]
    assert str(roots[0].resolve()) not in generated[0]
    assert str(roots[1].resolve()) not in generated[1]

    output_prefix = relative_build_dirs[0].as_posix()
    assert f"builddir = {output_prefix}/.ninja" in generated[0]
    assert f"build {output_prefix}/sources_kernel.cuda.o:" in generated[0]
    assert f"build {output_prefix}/test_module.so:" in generated[0]


def test_root_aot_subninja_manifest_parses(monkeypatch, tmp_path):
    monkeypatch.setattr(cpp_ext, "get_cuda_path", lambda: "/usr/local/cuda")
    monkeypatch.setattr(cpp_ext, "get_cuda_version", lambda: Version("13.0"))
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "8.9")
    root = tmp_path / "jit"
    monkeypatch.setattr(core.jit_env, "FLASHINFER_JIT_DIR", root)
    source = tmp_path / "sources" / "kernel.cu"
    specs = [
        _test_jit_spec("first_module", source),
        _test_jit_spec("second_module", source),
    ]

    for spec in specs:
        spec.write_ninja()

    aggregate = root / "aggregate.ninja"
    aggregate.write_text(
        "ninja_required_version = 1.3\n"
        + "".join(f"subninja {spec.ninja_path}\n" for spec in specs)
    )
    parsed = subprocess.run(
        ["ninja", "-f", str(aggregate), "-t", "targets"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert parsed.returncode == 0, parsed.stderr
    for spec in specs:
        assert str(spec.jit_library_path.relative_to(root)) in parsed.stdout


def test_copy_built_kernels_uses_digest_scoped_source(monkeypatch, tmp_path):
    root = tmp_path / "jit"
    monkeypatch.setattr(core.jit_env, "FLASHINFER_JIT_DIR", root)
    spec = _test_jit_spec("test_module", tmp_path / "sources" / "kernel.cu")
    monkeypatch.setattr(spec, "_ninja_build_digest", lambda: "test-digest")
    spec.jit_library_path.parent.mkdir(parents=True)
    spec.jit_library_path.write_bytes(b"test-library")

    out = tmp_path / "out"
    aot.copy_built_kernels([spec], out)

    assert (out / "test_module" / "test_module.so").read_bytes() == b"test-library"


def test_customize_batch_prefill_nvfp4_large_head_uses_prefill_flags(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)
    monkeypatch.setattr(
        attention_modules.current_compilation_context, "TARGET_CUDA_ARCHS", {(8, 6)}
    )
    monkeypatch.setattr(
        attention_modules.jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path / "gen"
    )

    spec = attention_modules.gen_customize_batch_prefill_module(
        "fa2",
        "test_batch_prefill_nvfp4_large_head",
        torch.float16,
        torch.uint8,
        torch.float16,
        torch.int32,
        512,
        512,
        [],
        [],
        ["sm_scale"],
        ["double"],
        "DefaultAttention<false, false, false, false>",
        "#include <flashinfer/attention/variants.cuh>",
    )

    assert any("sm_86" in flag for flag in spec.extra_cuda_cflags)
    with pytest.raises(RuntimeError, match="No supported CUDA architectures"):
        attention_modules._fa2_head_dim_nvcc_flags(512, 512, torch.uint8)


@pytest.mark.parametrize(
    ("head_dim_qk", "head_dim_vo", "supported"),
    [
        (64, 64, True),
        (128, 128, True),
        (256, 256, True),
        (192, 128, True),
        (512, 512, False),
        (256, 128, False),
        (128, 192, False),
    ],
)
def test_fa3_prefill_head_dim_supported(head_dim_qk, head_dim_vo, supported):
    assert is_fa3_prefill_head_dim_supported(head_dim_qk, head_dim_vo) is supported


@pytest.mark.parametrize(
    ("head_dim_qk", "head_dim_vo", "expected_backend"),
    [
        (256, 256, "fa3"),
        (192, 128, "fa3"),
        (512, 512, "fa2"),
    ],
)
def test_determine_attention_backend_respects_fa3_prefill_head_dim(
    monkeypatch, head_dim_qk, head_dim_vo, expected_backend
):
    monkeypatch.setattr(flashinfer.utils, "is_sm90a_supported", lambda device: True)

    backend = determine_attention_backend(
        torch.device("cuda"),
        PosEncodingMode.NONE.value,
        use_fp16_qk_reductions=False,
        use_custom_mask=False,
        dtype_q=torch.float16,
        dtype_kv=torch.float16,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
    )

    assert backend == expected_backend


def test_prefill_jit_helper_skips_fa3_unsupported_large_head(monkeypatch):
    calls = []

    def fake_single_prefill_module(
        backend,
        dtype_q,
        dtype_kv,
        dtype_o,
        head_dim_qk,
        head_dim_vo,
        *_args,
    ):
        calls.append(("single", backend, head_dim_qk, head_dim_vo))
        return SimpleNamespace(name=f"{backend}_single_{head_dim_qk}_{head_dim_vo}")

    def fake_batch_prefill_module(
        backend,
        dtype_q,
        dtype_kv,
        dtype_o,
        idtype,
        head_dim_qk,
        head_dim_vo,
        *_args,
    ):
        calls.append(("batch", backend, head_dim_qk, head_dim_vo))
        return SimpleNamespace(name=f"{backend}_batch_{head_dim_qk}_{head_dim_vo}")

    monkeypatch.setattr(jit_utils, "is_sm90a_supported", lambda device: True)
    monkeypatch.setattr(
        flashinfer.prefill, "gen_single_prefill_module", fake_single_prefill_module
    )
    monkeypatch.setattr(
        flashinfer.prefill, "gen_batch_prefill_module", fake_batch_prefill_module
    )
    monkeypatch.setattr(
        flashinfer.quantization,
        "gen_quantization_module",
        lambda: SimpleNamespace(name="quantization"),
    )
    monkeypatch.setattr(
        flashinfer.page,
        "gen_page_module",
        lambda: SimpleNamespace(name="page"),
    )

    jit_utils.gen_prefill_attention_modules(
        q_dtypes=[torch.float16],
        kv_dtypes=[torch.float16],
        head_dims=[512],
        pos_encoding_modes=[PosEncodingMode.NONE.value],
        use_sliding_window_options=[False],
        use_logits_soft_cap_options=[False],
        use_fp16_qk_reduction_options=[False],
    )

    assert ("single", "fa3", 512, 512) not in calls
    assert ("batch", "fa3", 512, 512) not in calls
    assert ("single", "fa2", 512, 512) in calls
    assert ("batch", "fa2", 512, 512) in calls
