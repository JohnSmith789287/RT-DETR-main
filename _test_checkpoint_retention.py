"""Smoke tests for detection checkpoint retention policy."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rtdetrv2_pytorch'))

from src.solver.det_solver import (
    _checkpoint_paths_for_epoch,
    _cleanup_periodic_checkpoints,
)


def test_periodic_checkpoint_paths():
    output_dir = Path('output')
    saved = []
    for epoch in range(40):
        paths = _checkpoint_paths_for_epoch(output_dir, epoch, checkpoint_freq=10)
        assert paths[0] == output_dir / 'last.pth'
        saved.extend(path.name for path in paths[1:])

    assert saved == [
        'checkpoint0009.pth',
        'checkpoint0019.pth',
        'checkpoint0029.pth',
        'checkpoint0039.pth',
    ]


def test_cleanup_keeps_last_three_periodic_checkpoints_only():
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        for name in [
            'checkpoint0009.pth',
            'checkpoint0019.pth',
            'checkpoint0029.pth',
            'checkpoint0039.pth',
            'checkpoint.pth',
            'last.pth',
            'best.pth',
        ]:
            (output_dir / name).write_text(name)

        removed = _cleanup_periodic_checkpoints(output_dir, max_keep=3)

        assert [path.name for path in removed] == ['checkpoint0009.pth']
        assert not (output_dir / 'checkpoint0009.pth').exists()
        assert (output_dir / 'checkpoint0019.pth').exists()
        assert (output_dir / 'checkpoint0029.pth').exists()
        assert (output_dir / 'checkpoint0039.pth').exists()
        assert (output_dir / 'checkpoint.pth').exists()
        assert (output_dir / 'last.pth').exists()
        assert (output_dir / 'best.pth').exists()


if __name__ == '__main__':
    test_periodic_checkpoint_paths()
    test_cleanup_keeps_last_three_periodic_checkpoints_only()
    print('ALL CHECKPOINT RETENTION TESTS PASSED')
