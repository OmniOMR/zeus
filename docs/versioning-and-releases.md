# Versioning and releases

Zeus is one package in one repository, so a release is one git tag and there is only one version number to think about — unlike [Musibot](https://github.com/OmniOMR/musibot/blob/main/docs/versioning-and-releases.md), whose components are released independently and whose tags therefore have to name one.

What Zeus does share with Musibot is the important half: **the version is not written down.** It is derived from the git tags at build time.


## The version comes from the tags

`pyproject.toml` declares the version *dynamic* and lets [hatch-vcs](https://github.com/ofek/hatch-vcs) derive it:

```toml
[build-system]
requires = ["hatchling", "hatch-vcs"]

[project]
name = "zeus"
dynamic = ["version"]

[tool.hatch.version]
source = "vcs"
```

That is the whole configuration. Musibot's components each need `root`, `git_describe_command` and `tag_regex` on top of this, because they are subdirectories of a monorepo and must read only their own tags; Zeus is at the repository root and has only its own, and setuptools-scm strips a leading `v` by itself.

What you get out:

| Where you are | Version built |
| --- | --- |
| Exactly on `v1.0.0`, clean tree | `1.0.0` |
| 3 commits after it | `1.0.1.dev3+g2d478f1` |
| 3 commits after it, with uncommitted edits | `1.0.1.dev3+g2d478f1.d20260728` |
| Before the first tag | `0.1.dev24+g511d939` |

The `.dev` number counts commits since the tag, so it rises monotonically and every commit gets a distinct, correctly-ordered version. The `1.0.1` part is a *guess* — setuptools-scm assumes the next release bumps the patch. It is not a promise about what the next release will be numbered; only the ordering matters, and a development build is not a release.

A dirty working tree counts as being past the tag, which is why the second and third rows differ. Building a release from a tree with uncommitted changes therefore cannot silently produce a release version.


### Why not just write the version down

Because `pip` decides whether to reinstall by comparing **version strings, and nothing else**. It records the commit it installed from in `direct_url.json` and then ignores it when making that decision. With a hand-maintained version that stays `1.0.0` across a development cycle:

| Situation | What pip does |
| --- | --- |
| Same version, new commit, plain install | **nothing, silently** |
| Same version, new commit, `pip install -U` | **nothing, silently** — `-U` does not help |
| Different version, plain install | installs it, no `-U` needed |
| Different version, older commit | downgrades — it syncs to whatever the URL builds |
| `--force-reinstall --no-deps` | always reinstalls |

The first two rows are the trap, and Zeus is installed from a git URL, which is exactly where it bites: a colleague reinstalls from a newer commit, pip prints nothing alarming, and they keep running the old code. Deriving the version from git turns every commit into a distinct version, which lands you in row three — the one where things simply work.


## Three numbers that are easy to confuse

A Zeus release involves three version-shaped things, and they move independently.

| Number | What it versions | Where it lives |
| --- | --- | --- |
| **The package version** | The code: the CLI, the python API, the snapshot format. Semver, derived from the tags as above. | Nowhere — computed at build time from `git describe`. |
| **The Musibot model version** | One trained snapshot. It is what a *Pipeline* pins and what [discovery](https://github.com/OmniOMR/musibot/blob/main/docs/discovery.md) announces; Musibot treats it as an opaque string and never parses it. | `musibot_model_version` in the snapshot's [`model_options.yaml`](model-snapshots.md). |
| **`ipc_version`** | The [worker IPC contract](musibot-model.md) Zeus speaks. One integer, `1` today, checked by a worker head for exact equality. | `zeus.musibot.protocol.IPC_VERSION`. |

The first two are deliberately independent. A snapshot trained in July and a Zeus released in December are different things with different lifetimes, and what a *Pipeline* pinned must not change because the package was repackaged. Publishing a new Zeus version does not re-version anybody's models.

`ipc_version` moves only when the wire contract does, which is rarely; see [Musibot's rules for when to bump it](https://github.com/OmniOMR/musibot/blob/main/docs/worker-ipc.md#when-to-bump-it).


## Installing a released version

```bash
pip install 'zeus @ git+https://github.com/OmniOMR/zeus.git@v1.0.0'
```

To follow development rather than a release, put a branch name or a commit SHA where the tag goes. Because each commit builds a distinct version, a plain `pip install` of a newer commit replaces what is there — no `--force-reinstall` needed.

Zeus requires **python 3.10** and will refuse to install on anything else; see [the README](../README.md#usage) for why, and [the Musibot integration](musibot-model.md#deployment-two-virtual-environments) for what that means when deploying.


## Cutting a release

Say Zeus is going to `1.0.0`.

1. Move `CHANGELOG.md` entries from *Unreleased* into a `## 1.0.0 — 2026-07-28` section.
2. Commit that, and anything else the release needs. There is **no version to bump** — that is the point of deriving it.
3. Tag and push:

   ```bash
   git tag v1.0.0
   git push origin main v1.0.0
   ```

4. Create the release page, with that changelog section as its notes:

   ```bash
   gh release create v1.0.0 --title "Zeus 1.0.0" --notes-file notes.md
   ```

5. Attach the model snapshots this release publishes as release assets — see [Model snapshots](model-snapshots.md).

6. Sanity-check that the tag builds clean, with no `.dev` suffix:

   ```bash
   pip download --no-deps -d /tmp/check \
     'zeus @ git+https://github.com/OmniOMR/zeus.git@v1.0.0'
   ```


## When the build needs git

The version is computed at build time by running `git`, so the build needs a repository:

- **A source archive has no history.** GitHub's "Download ZIP", `git archive`, or a `COPY` into a Docker image that excludes `.git` all fail to build with `unable to detect version`. Installing from a `git+https://` link is fine — pip clones, so the history is there.
- **It needs the tags, not just the history.** A normal `git clone` fetches them and is fine; a shallow or `--no-tags` clone, typical in CI, is not. `git fetch --tags` fixes it. Without tags Zeus builds as something like `0.1.dev24+g511d939`, which is not wrong, only uninformative.
- **The escape hatch is `SETUPTOOLS_SCM_PRETEND_VERSION`**, which forces a version when there is no repository to read. The more precise `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_<NAME>` form does *not* work under hatch-vcs, which does not pass the distribution name through; use the plain variable.
- **The `+g<sha>` local segment cannot be uploaded to PyPI.** It never appears on a tagged commit, so it is not a problem, but it is the thing to remember on the day Zeus is published to an index.


## The changelog

[CHANGELOG.md](../CHANGELOG.md) is in [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) form: an *Unreleased* section at the top that accumulates entries as work lands, and one section per released version below it. It is the source text for the GitHub release notes, and it is the only human-written record of what a version contains — the tag says when, the changelog says what.

Entries are written for whoever installs Zeus, so they describe behaviour and contracts rather than commits.
