# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check the actual decoder layout using symbol sizes and ELF load-segment ranges."""

import os
import subprocess
import sys


def main(nm, readelf, image):
    def inspect(tool, *options):
        return subprocess.check_output(
            [tool, *options, image], text=True, env={**os.environ, 'LC_ALL': 'C'}
        ).splitlines()

    scope = 'rocjitsu::cdna5::DecoderImpl::'
    table = scope + 'primary_decode_table'
    relative = 'rj_test_expect_relative_decode_table'
    switch = 'rj_test_expect_decode_switch'
    symbols = {}
    for line in inspect(nm, '-C', '-S', '--format=posix'):
        fields = line.split()
        if fields and (
            fields[0] in (table, relative, switch)
            or (fields[0].startswith(scope + 'sub_decode_') and '(' not in fields[0])
        ):
            name, kind, address, size = fields
            symbols[name] = kind, int(address, 16), int(size, 16)

    if (relative in symbols) == (switch in symbols):
        raise AssertionError('missing or ambiguous decoder representation marker')
    if switch in symbols:
        retained = sorted(symbols.keys() - {switch})
        if retained:
            raise AssertionError(f'switch dispatch unexpectedly retained {retained}')
        return
    if table not in symbols:
        raise AssertionError(f'missing {table}')
    kind, start, size = symbols[table]
    if size != 512 * 4:
        raise AssertionError(
            f'{table} has {size} bytes, expected 512 signed 32-bit offsets'
        )
    if kind not in ('R', 'r'):
        raise AssertionError(f'{table} is not a read-only data symbol')

    headers = inspect(readelf, '-lWd')
    for line in headers:
        fields = line.split()
        if fields and fields[0] == 'LOAD':
            segment, length = int(fields[2], 16), int(fields[5], 16)
            if segment <= start and start + size <= segment + length:
                if 'W' in ''.join(fields[6:-1]):
                    raise AssertionError(
                        f'{table} is in a writable load segment: {line}'
                    )
                break
    else:
        raise AssertionError(f'{table} does not fit in a load segment')

    # Without text relocations the loader never writes to this segment.
    if any('TEXTREL' in line for line in headers):
        raise AssertionError(
            'the model-only image permits loader writes to read-only segments'
        )


if __name__ == '__main__':
    main(*sys.argv[1:])
