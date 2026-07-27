"""Zeus — an image-to-sequence model for Optical Music Recognition.

Zeus reads an image of a staff or a grandstaff and produces the music notation
on it, as LMX tokens and from those as MusicXML. The package is both a library
and the `zeus` command line tool; see README.md and the docs/ folder.

Nothing is imported here on purpose. TensorFlow takes seconds to import, and
`zeus --help` should not pay for it, so every entry point defers its heavy
imports until it knows it needs them.
"""
