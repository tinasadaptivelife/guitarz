"""Behavioral spec for the guitarz backend (``/api/browse``, ``/api/file``,
``/api/soundfonts``, static).

These tests pin down the security-relevant contract of the file API: only
Guitar Pro files inside the configured root are ever visible or downloadable,
regardless of how creatively a request tries to point elsewhere (relative
traversal, absolute paths, symlinks, control characters).
"""

from pathlib import Path

from fastapi.testclient import TestClient

from guitarz.server import create_app


class TestBrowse:
    """Directory-listing behavior of ``GET /api/browse``."""

    def test_root_listing_shows_allowed_files_and_dirs_only(self, client):
        """The root listing exposes GP files and visible folders — nothing else.

        ``README.md`` (disallowed extension), ``.hidden.gp`` (hidden file) and
        ``.hiddendir`` (hidden folder) must all be absent even though they sit
        in the root directory.
        """
        data = client.get("/api/browse", params={"path": ""}).json()
        assert data["path"] == ""
        assert [d["name"] for d in data["dirs"]] == ["band"]
        assert [f["name"] for f in data["files"]] == ["song one.gp"]

    def test_subfolder_listing_filters_extensions_and_sorts(self, client):
        """A subfolder lists GP files of every generation, sorted, txt excluded.

        The allowlist covers .gp/.gp3/.gp4/.gp5/.gpx — alphaTab reads all of them,
        and collections routinely mix eras. Sorting is part of the contract
        because the UI renders entries in the order the API returns them.
        """
        data = client.get("/api/browse", params={"path": "band"}).json()
        assert data["path"] == "band"
        assert [d["name"] for d in data["dirs"]] == ["nested"]
        assert [f["name"] for f in data["files"]] == [
            "Alpha song.gp5",
            "old.gp3",
            "older.gp4",
            "riff.gp5",
            "six.gpx",
        ]

    def test_legacy_gp3_gp4_are_downloadable(self, client):
        """GP3 and GP4 files are served like any other allowed format."""
        assert client.get("/api/file", params={"path": "band/old.gp3"}).content == b"GP3-OLD"
        assert client.get("/api/file", params={"path": "band/older.gp4"}).content == b"GP4-OLDER"

    def test_gpx_is_downloadable(self, client):
        """Guitar Pro 6 (.gpx) files are served like any other allowed format.

        Free tab sites hand out many .gpx files and alphaTab parses them, so
        hiding them would leave downloaded songs invisible in the explorer.
        """
        res = client.get("/api/file", params={"path": "band/six.gpx"})
        assert res.status_code == 200
        assert res.content == b"GPX-SIX"

    def test_file_entries_report_relative_path_and_size(self, client):
        """File entries carry the root-relative POSIX path and byte size.

        The frontend passes ``path`` straight back to ``/api/file``, so it must
        be relative to the root (never absolute) and use forward slashes.
        """
        data = client.get("/api/browse", params={"path": "band"}).json()
        riff = next(f for f in data["files"] if f["name"] == "riff.gp5")
        assert riff["path"] == "band/riff.gp5"
        assert riff["size"] == len(b"GP5-RIFF")

    def test_missing_directory_is_404(self, client):
        """Browsing a nonexistent directory returns 404, not a server error."""
        assert client.get("/api/browse", params={"path": "no-such-dir"}).status_code == 404

    def test_browsing_a_file_is_404(self, client):
        """Browsing a path that is a file (not a directory) returns 404."""
        assert client.get("/api/browse", params={"path": "song one.gp"}).status_code == 404

    def test_relative_traversal_is_403(self, client):
        """``..`` segments that escape the root are rejected with 403."""
        assert client.get("/api/browse", params={"path": "../.."}).status_code == 403

    def test_absolute_path_is_403(self, client):
        """An absolute path is rejected: it replaces the root when joined."""
        assert client.get("/api/browse", params={"path": "/etc"}).status_code == 403


class TestFileDownload:
    """Byte-serving behavior of ``GET /api/file``."""

    def test_download_is_byte_identical(self, client):
        """A served file's bytes match the file on disk exactly.

        alphaTab parses the raw bytes in the browser, so any transformation
        (encoding, newline mangling) would corrupt the score.
        """
        res = client.get("/api/file", params={"path": "song one.gp"})
        assert res.status_code == 200
        assert res.content == b"GP-BYTES-ROOT"
        assert res.headers["content-type"] == "application/octet-stream"

    def test_download_from_subfolder(self, client):
        """Files inside subfolders are served via their relative path."""
        res = client.get("/api/file", params={"path": "band/riff.gp5"})
        assert res.status_code == 200
        assert res.content == b"GP5-RIFF"

    def test_disallowed_extension_is_404(self, client):
        """A file with a non-GP extension is never served, even if it exists.

        This keeps the endpoint from becoming a generic file-exfiltration
        primitive should the root ever contain non-tab material.
        """
        assert client.get("/api/file", params={"path": "band/notes.txt"}).status_code == 404

    def test_missing_file_is_404(self, client):
        """Requesting a nonexistent file returns 404."""
        assert client.get("/api/file", params={"path": "ghost.gp"}).status_code == 404

    def test_directory_is_404(self, client):
        """Requesting a directory from the file endpoint returns 404."""
        assert client.get("/api/file", params={"path": "band"}).status_code == 404

    def test_relative_traversal_is_403(self, client):
        """``..`` traversal out of the root is rejected with 403."""
        assert client.get("/api/file", params={"path": "../../etc/passwd"}).status_code == 403

    def test_absolute_path_is_403(self, client):
        """Absolute paths are rejected with 403."""
        assert client.get("/api/file", params={"path": "/etc/passwd"}).status_code == 403


class TestPathHardening:
    """Hostile-input cases that must degrade to clean 4xx responses."""

    def test_null_byte_is_client_error_not_500(self, client):
        """A NUL byte in the path yields a 4xx, never an unhandled 500.

        ``Path.resolve`` raises ``ValueError`` on embedded null bytes; the API
        must translate that into a client error instead of crashing.
        """
        res = client.get("/api/file", params={"path": "song\x00.gp"})
        assert 400 <= res.status_code < 500

    def test_symlink_escaping_root_is_403(self, client, music_root, tmp_path_factory):
        """A symlink inside the root pointing outside it is not followed.

        ``resolve()`` canonicalizes symlinks, so a link to an outside directory
        must fail the containment check even though its own path looks safe.
        """
        outside = tmp_path_factory.mktemp("outside")
        (outside / "leak.gp").write_bytes(b"outside-bytes")
        (music_root / "link").symlink_to(outside)
        assert client.get("/api/file", params={"path": "link/leak.gp"}).status_code == 403
        assert client.get("/api/browse", params={"path": "link"}).status_code == 403


class TestStaticFrontend:
    """The SPA is served from the root path."""

    def test_index_served_at_root(self, client):
        """``GET /`` returns the app shell so the SPA loads without a path."""
        res = client.get("/")
        assert res.status_code == 200
        assert "guitarz" in res.text


class TestSoundfonts:
    """Catalog contract of ``GET /api/soundfonts`` (the transport's SoundFont menu).

    The endpoint scans a SoundFont directory rather than returning a hardcoded
    list, so a font dropped into the folder shows up in the menu without a code
    change. Each test builds its own directory and injects it through
    ``create_app(..., soundfont_dir=...)`` so the real vendored fonts never
    influence the result.
    """

    @staticmethod
    def _client(music_root: Path, sf_dir: Path, *names: str) -> TestClient:
        """Create ``names`` as dummy files in ``sf_dir`` and return a client for it.

        SoundFont contents are irrelevant to the catalog (it never opens the
        files), so a few placeholder bytes stand in for multi-megabyte fonts.

        Args:
            music_root: disposable music root for ``create_app``.
            sf_dir: directory to use as the SoundFont folder (created here).
            *names: file names to create inside ``sf_dir``.

        Returns:
            A ``TestClient`` for an app whose SoundFont folder is ``sf_dir``.
        """
        sf_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            (sf_dir / name).write_bytes(b"RIFF-sfbk")
        return TestClient(create_app(music_root, soundfont_dir=sf_dir))

    def test_lists_only_sf2_and_sf3_sorted_by_label(self, music_root, tmp_path):
        """Only visible .sf2/.sf3 files are listed, sorted case-insensitively by label.

        LICENSE files ship next to the fonts and hidden files may appear
        (editor or OS droppings); neither may show up as a playable choice.
        """
        client = self._client(
            music_root, tmp_path / "sf",
            "sonivox.sf2", "MuseScore_General.sf3", "GeneralUser-GS.sf2",
            "LICENSE", "notes.txt", ".hidden.sf2",
        )
        data = client.get("/api/soundfonts").json()
        assert [f["file"] for f in data["fonts"]] == [
            "GeneralUser-GS.sf2",
            "MuseScore_General.sf3",
            "sonivox.sf2",
        ]

    def test_known_fonts_get_friendly_labels_others_use_stem(self, music_root, tmp_path):
        """Bundled fonts show curated names; unknown fonts fall back to their file stem."""
        client = self._client(
            music_root, tmp_path / "sf",
            "MuseScore_General.sf3", "GeneralUser-GS.sf2", "sonivox.sf2", "MyFont.sf3",
        )
        labels = {f["file"]: f["label"] for f in client.get("/api/soundfonts").json()["fonts"]}
        assert labels == {
            "MuseScore_General.sf3": "MuseScore General",
            "GeneralUser-GS.sf2": "GeneralUser GS",
            "sonivox.sf2": "Sonivox (light)",
            "MyFont.sf3": "MyFont",
        }

    def test_url_points_at_static_vendor_path(self, music_root, tmp_path):
        """Each font's ``url`` is its path under the static mount, URL-encoded.

        The frontend hands this string straight to alphaTab's loader, so it
        must be fetchable as-is — including names containing spaces.
        """
        client = self._client(music_root, tmp_path / "sf", "sonivox.sf2", "My Font.sf2")
        urls = {f["file"]: f["url"] for f in client.get("/api/soundfonts").json()["fonts"]}
        assert urls == {
            "sonivox.sf2": "/vendor/alphatab/soundfont/sonivox.sf2",
            "My Font.sf2": "/vendor/alphatab/soundfont/My%20Font.sf2",
        }

    def test_default_prefers_musescore(self, music_root, tmp_path):
        """MuseScore General is the default whenever it is installed."""
        client = self._client(
            music_root, tmp_path / "sf",
            "sonivox.sf2", "MuseScore_General.sf3", "GeneralUser-GS.sf2",
        )
        assert client.get("/api/soundfonts").json()["default"] == "MuseScore_General.sf3"

    def test_default_falls_back_to_sonivox(self, music_root, tmp_path):
        """Without MuseScore, the always-vendored sonivox font is the default."""
        client = self._client(music_root, tmp_path / "sf", "GeneralUser-GS.sf2", "sonivox.sf2")
        assert client.get("/api/soundfonts").json()["default"] == "sonivox.sf2"

    def test_default_falls_back_to_first_font(self, music_root, tmp_path):
        """With no preferred font present, the first listed font is the default."""
        client = self._client(music_root, tmp_path / "sf", "Zeta.sf2", "Alpha.sf3")
        assert client.get("/api/soundfonts").json()["default"] == "Alpha.sf3"

    def test_missing_directory_yields_empty_catalog(self, music_root, tmp_path):
        """A missing SoundFont folder (vendor not fetched) is an empty list, not a 500."""
        client = TestClient(create_app(music_root, soundfont_dir=tmp_path / "absent"))
        res = client.get("/api/soundfonts")
        assert res.status_code == 200
        assert res.json() == {"default": None, "fonts": []}


class TestCacheHeaders:
    """Freshness contract: browsers must revalidate every asset.

    Regression suite for a real field bug: after a UI feature shipped, the
    browser kept serving months-stale cached JavaScript because responses
    carried no ``Cache-Control`` header, leaving freshness to browser
    heuristics — new HTML rendered a button wired to nothing. ``no-cache``
    forces conditional requests (ETag/304), which keeps repeat loads cheap
    while guaranteeing that a server restart is enough to ship new frontend
    code to every open browser.
    """

    def test_static_and_api_responses_demand_revalidation(self, client):
        """Every response type carries Cache-Control: no-cache."""
        for path in ("/", "/css/style.css", "/js/transport.js", "/api/browse"):
            res = client.get(path)
            assert res.status_code == 200, path
            assert res.headers.get("cache-control") == "no-cache", (
                f"{path} must demand revalidation, got: "
                f"{res.headers.get('cache-control')!r}"
            )
