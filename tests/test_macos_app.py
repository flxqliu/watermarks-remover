"""Guards on the macOS app wrapper (app/macos).

The Swift front end is a thin client over the same HTTP service the skill
drives, so the things that can silently break it are contract drifts: an option
the server stops accepting, an endpoint that moves, a bundled module that stops
being copied. Swift does not compile here, so these tests read the sources as
text and check the contract instead.
"""

from __future__ import annotations

import ast
import plistlib
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
APP = ROOT / "app" / "macos"
sys.path.insert(0, str(SCRIPTS))

import server

pytestmark = pytest.mark.skipif(not APP.exists(), reason="macOS app not present")


def _swift_source(*parts: str) -> str:
    return (APP.joinpath(*parts)).read_text(encoding="utf-8")


def test_info_plist_matches_the_swift_product() -> None:
    with (APP / "Info.plist").open("rb") as handle:
        info = plistlib.load(handle)

    package = _swift_source("Package.swift")
    assert 'name: "WatermarksRemover"' in package

    assert info["CFBundleExecutable"] == "WatermarksRemover"
    assert info["CFBundlePackageType"] == "APPL"
    assert info["CFBundleIconFile"] == "AppIcon"
    assert info["LSMinimumSystemVersion"] == "13.0"
    assert info["CFBundleIdentifier"].startswith("io.github.")


def test_clean_options_are_all_accepted_by_the_server() -> None:
    """Every option key the app sends must be one the server allows."""
    source = _swift_source("Sources", "WatermarksRemover", "Model", "CleanOptions.swift")
    body = source.split("var wireOptions:", 1)[1]
    sent = set(re.findall(r'options\["([a-z_]+)"\]|^\s+"([a-z_]+)":', body, re.MULTILINE))
    keys = {found for pair in sent for found in pair if found}

    assert keys, "no option keys parsed out of CleanOptions.swift"
    unknown = keys - set(server.ALLOWED_CLEAN_OPTIONS)
    assert not unknown, f"app sends options the server rejects: {sorted(unknown)}"

    for key in keys:
        expected = server.ALLOWED_CLEAN_OPTIONS[key]
        wire = "string" if expected is str else "bool"
        assert f'"{key}"' in body
        # remove_pixel is the only string-valued option; the rest are booleans.
        assert (key == "remove_pixel") == (wire == "string")


def test_pixel_remover_values_match_the_server() -> None:
    source = _swift_source("Sources", "WatermarksRemover", "Model", "CleanOptions.swift")
    cases = set(re.findall(r"^\s+case (ctrlregen|diffusion)$", source, re.MULTILINE))
    assert cases == {"ctrlregen", "diffusion"}

    handler = (SCRIPTS / "server.py").read_text(encoding="utf-8")
    assert 'remove_pixel not in (None, "ctrlregen", "diffusion")' in handler


def test_client_only_calls_endpoints_the_server_routes() -> None:
    source = _swift_source("Sources", "WatermarksRemover", "Service", "ServiceClient.swift")
    called = set(re.findall(r'await (?:get|post)\("(/[a-z/]+)"', source))
    assert called, "no endpoints parsed out of ServiceClient.swift"

    handler = (SCRIPTS / "server.py").read_text(encoding="utf-8")
    for path in called:
        assert f'"{path}"' in handler, f"{path} is not routed by server.py"


def test_input_cap_matches_the_service() -> None:
    source = _swift_source("Sources", "WatermarksRemover", "Service", "ServiceClient.swift")
    match = re.search(r"maxInputBytes = (\d+) << (\d+)", source)
    assert match, "maxInputBytes not found"
    app_cap = int(match.group(1)) << int(match.group(2))

    from common import MAX_INPUT_BYTES

    assert app_cap == MAX_INPUT_BYTES


def test_api_key_is_passed_through_the_environment_not_argv() -> None:
    """The server reads WATERMARKS_SERVER_API_KEY, which keeps it out of `ps`."""
    controller = _swift_source("Sources", "WatermarksRemover", "Service", "ServiceController.swift")
    assert 'environment["WATERMARKS_SERVER_API_KEY"] = apiKey' in controller
    assert "--api-key" not in controller
    assert 'API_KEY = os.environ.get("WATERMARKS_SERVER_API_KEY"' in (
        SCRIPTS / "server.py"
    ).read_text(encoding="utf-8")


def test_bundle_ships_every_module_the_service_imports() -> None:
    """build-app.sh copies service/scripts/*.py — check that this is enough."""
    build = _swift_source("Scripts", "build-app.sh")
    assert 'cp "$repo"/service/scripts/*.py' in build

    tree = ast.parse((SCRIPTS / "server.py").read_text(encoding="utf-8"))
    local_modules = {path.stem for path in SCRIPTS.glob("*.py")}
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)

    for module in imported & local_modules:
        assert (SCRIPTS / f"{module}.py").exists()
    # The server's own siblings must be plain modules, not packages, or the
    # flat *.py copy in build-app.sh would miss them.
    assert not [path for path in SCRIPTS.iterdir() if path.is_dir() and path.name != "__pycache__"]


def test_icon_generator_writes_a_full_iconset(tmp_path: Path) -> None:
    sys.path.insert(0, str(APP / "Scripts"))
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_icon", APP / "Scripts" / "make_icon.py")
    assert spec and spec.loader
    make_icon = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_icon)

    out = tmp_path / "AppIcon.iconset"
    out.mkdir()
    master = make_icon.render(64)
    for name, size in make_icon.ICONSET_SIZES:
        if size > 64:
            continue
        make_icon.write_png(out / name, make_icon.downsample(master, 64, size), size)
        written = (out / name).read_bytes()
        assert written.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(written) > 100
