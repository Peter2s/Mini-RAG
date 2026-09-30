import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.enums import FileExtension


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("notes.txt", FileExtension.TXT),
        ("README.MD", FileExtension.MD),
        ("dir/report.Pdf", FileExtension.PDF),
        ("archive.tar.gz", None),
        ("no_extension", None),
    ],
)
def test_from_path(name, expected):
    assert FileExtension.from_path(name) is expected


def test_settings_default_allows_every_extension():
    assert Settings(_env_file=None).allowed_extensions == list(FileExtension)


def test_settings_parse_and_validate_extensions():
    settings = Settings(_env_file=None, allowed_extensions=[".txt", ".pdf"])
    assert settings.allowed_extensions == [FileExtension.TXT, FileExtension.PDF]
    with pytest.raises(ValidationError):
        Settings(_env_file=None, allowed_extensions=[".exe"])
