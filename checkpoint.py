"""Checkpoint and resume, without changing what a run produces.

A resumed run must be bit-identical to an uninterrupted one. Two things are
kept separate:

  STATE      everything the loop has accumulated -- accumulator, raw counts,
             batch index, readout-bit count, photon counters, and for the
             adaptive driver the pixel histories, hierarchy state and
             partition trace. Plain array data, saved and loaded exactly.

  POSITION   where the RNG streams are. cupy generators can't be pickled or
             seeked, so position isn't saved; reposition_streams below
             re-establishes it by redrawing and discarding.

Writes are atomic (temp file then rename), so a job killed mid-write leaves
the previous checkpoint intact. A checkpoint records a fingerprint of the
run's configuration and refuses to load into a different one.

Resuming is O(resume point): both the photon sampler and the code families
carry generator state, so resuming at batch 340k means redrawing 340k
batches first.
"""
import os

import numpy as np

import codes


CHECKPOINT_VERSION = 1


def fingerprint(config_items):
    """Stable string identifying a run configuration, independent of dict
    insertion order -- e.g. 'batch_size=100|g_h=3|g_w=4|scene=...'.
    config_items should include anything that alters the accumulated
    numbers, and exclude post-processing-only settings (bilateral params).
    """
    parts = [f'{key}={config_items[key]}' for key in sorted(config_items)]
    return '|'.join(parts)


def path_for(result_path):
    """Checkpoint path derived from a run's output path: a dotfile beside
    it, e.g. '/dir/.checkpoint_<name>.npz'.
    """
    directory = os.path.dirname(result_path)
    name = os.path.basename(result_path)
    return os.path.join(directory, f'.checkpoint_{name}')


def save(path, next_batch, config_items, arrays):
    """Write a checkpoint atomically.

    Args:
        path:         destination, from path_for.
        next_batch:   the batch a resumed run should start from (one past
                      the last completed batch).
        config_items: configuration dict, fingerprinted into the file.
        arrays:       dict of name -> numpy array (host arrays; caller must
                      cp.asnumpy device arrays first).

    Returns:
        str -- the path written.
    """
    payload = dict(
        checkpoint_version=np.int32(CHECKPOINT_VERSION),
        next_batch=np.int64(next_batch),
        config_fingerprint=np.array(fingerprint(config_items)),
    )
    for name, array in arrays.items():
        payload[name] = array

    temporary = f'{path}.tmp.npz'
    np.savez(temporary, **payload)
    os.replace(temporary, path)  # atomic
    return path


def load(path, config_items):
    """Read a checkpoint, or return None if there is none (the normal case
    for a fresh run).

    Raises:
        RuntimeError: if the file was written by a different configuration
            or checkpoint version -- loading it anyway would silently mix
            two experiments into one result.
    """
    if not os.path.exists(path):
        return None
    state = dict(np.load(path, allow_pickle=False))
    version = int(state['checkpoint_version'])
    if version != CHECKPOINT_VERSION:
        raise RuntimeError(
            f'checkpoint {path} is version {version}, this code writes '
            f'{CHECKPOINT_VERSION}; delete it or migrate it deliberately')
    saved = str(np.asarray(state['config_fingerprint']).item())
    expected = fingerprint(config_items)
    if saved != expected:
        raise RuntimeError(
            f'checkpoint {path} was written by a different configuration.\n'
            f'  checkpoint: {saved}\n  this run:   {expected}\n'
            'Resuming across configurations would silently mix two experiments; '
            'use a different OUT_TAG or delete the checkpoint.')
    return state


def reposition_streams(sampler, code_source, next_batch, labels_for_batch=None):
    """Advance both RNG streams to where an uninterrupted run would have
    left them, by drawing and discarding the first next_batch batches
    (cupy generators can't be seeked). The accumulated results already came
    back from the checkpoint; only stream position is restored here.

    Args:
        sampler:     a PhotonSampler, advanced in place.
        code_source: a CodeSource, advanced in place.
        next_batch:  the batch to resume from; 0 means nothing to do.
        labels_for_batch: callable(batch) -> (labels_flat, n_groups),
            required only for shared-code runs.

    Returns:
        int -- how many batches were replayed.
    """
    if next_batch <= 0:
        return 0
    sampler.skip_to(next_batch)
    code_source.fast_forward(next_batch, labels_for_batch=labels_for_batch)
    return next_batch


def discard(path):
    """Remove a checkpoint once the run has completed successfully. Missing
    is fine, not an error.
    """
    if os.path.exists(path):
        os.remove(path)
