from git import Repo
import re

repo = Repo('.')


def write_semver():
    new_version = get_semver()
    with open('Cargo.toml', 'r+') as f:
        text = f.read()
        text = re.sub(
            r'(?m)^version\s*=\s*".*?"',
            f'version = "{new_version}"',
            text,
            count=1
        )
        f.seek(0)
        f.write(text)
        f.truncate()


def get_semver():
    version = "0.0.0"
    commits = reversed(list(repo.iter_commits()))
    for c in commits:
        version_splitted = version.split(".")
        msg = c.message.lower()
        if msg.startswith("breaking:"):
            version = f"{int(version_splitted[0]) + 1}.0.0"
        elif msg.startswith("feat:"):
            version = f"{version_splitted[0]}.{int(version_splitted[1]) + 1}.0"
        elif msg.startswith("fix:") or msg.startswith("perf:"):
            version = f"{version_splitted[0]}.{version_splitted[1]}.{int(version_splitted[2]) + 1}"
    return version


if __name__ == '__main__':
    write_semver()
