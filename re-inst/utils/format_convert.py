#import sys

import numpy as np
import pretty_midi as pyd
import os
from scipy.interpolate import interp1d
from dataset import EMBED_PROGRAM_MAPPING, SLAKH_PROGRAM_MAPPING

def matrix2midi(matrices, programs, init_tempo=120, time_start=0):
        """
        Reconstruct a multi-track midi from a 3D matrix of shape (Track. Time, 128).
        """
        ACC = 16
        tracks = []
        for program in programs:
            track_recon = pyd.Instrument(program=int(program), is_drum=False, name=pyd.program_to_instrument_name(int(program)))
            tracks.append(track_recon)

        indices_track, indices_onset, indices_pitch = np.nonzero(matrices)
        alpha = 1 / (ACC // 4) * 60 / init_tempo #timetep between each quntization bin
        for idx in range(len(indices_track)):
            track_id = indices_track[idx]
            onset = indices_onset[idx]
            pitch = indices_pitch[idx]

            start = onset * alpha
            duration = matrices[track_id, onset, pitch] * alpha
            velocity = 100

            note_recon = pyd.Note(velocity=int(velocity), pitch=int(pitch), start=time_start + start, end=time_start + start + duration)
            tracks[track_id].notes.append(note_recon)
        
        midi_recon = pyd.PrettyMIDI(initial_tempo=init_tempo)
        midi_recon.instruments = tracks
        return midi_recon


def chord2midi(chord_table, init_tempo=120, time_start=0):
    chord_matrix = np.zeros((1, len(chord_table)*4, 128))
    last_chord = np.array([-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1])
    last_idx = -1
    for idx, chord in enumerate(chord_table):
        if chord[0] == -1:
            continue
        if (chord != last_chord).any():
            chroma = chord[1: -1]
            bass = int((chord[0] + chord[-1]) % 12)
            if bass < 6:
                chord_matrix[:, idx*4, 4*12+bass: 5*12+bass] = np.roll(chroma, shift=-int(bass)) * 4
            else:
                chord_matrix[:, idx*4, 3*12+bass: 4*12+bass] = np.roll(chroma, shift=-int(bass)) * 4
            last_chord = chord
            last_idx = idx
        else:
            chord_matrix[:, last_idx*4, :][chord_matrix[:, last_idx*4, :] > 0] += 4
    chord_recon = matrix2midi(chord_matrix, [0], init_tempo=init_tempo, time_start=time_start)
    return chord_recon


def grid2pr(grid, max_note_count=16, min_pitch=0, pitch_eos_ind=129):
        #grid: (time, max_simu_note, 6)
        if grid.shape[1] == max_note_count:
            grid = grid[:, 1:]
        pr = np.zeros((grid.shape[0], 128), dtype=int)
        for t in range(grid.shape[0]):
            for n in range(grid.shape[1]):
                note = grid[t, n]
                if note[0] == pitch_eos_ind:
                    break
                pitch = note[0] + min_pitch
                dur = int(''.join([str(_) for _ in note[1:]]), 2) + 1
                pr[t, pitch] = dur
        return pr


def midi2matrix(midi, quaver):
    pr_matrices = []
    programs = []
    for track in midi.instruments:
        programs.append(track.program)
        pr_matrix = np.zeros((len(quaver), 128))
        for note in track.notes:
            note_start = np.argmin(np.abs(quaver - note.start))
            note_end =  np.argmin(np.abs(quaver - note.end))
            if note_end == note_start:
                note_end = min(note_start + 1, len(quaver) - 1)
            pr_matrix[note_start, note.pitch] = note_end - note_start
        pr_matrices.append(pr_matrix)
    return np.array(pr_matrices), np.array(programs)


def midi2matrix_with_dynamics(midi, quaver):
        pr_matrices = []
        programs = []
        quantization_error = []
        for track in midi.instruments:
            qt_error = [] # record quantization error
            pr_matrix = np.zeros((len(quaver), 128, 2))
            for note in track.notes:
                note_start = np.argmin(np.abs(quaver - note.start))
                note_end =  np.argmin(np.abs(quaver - note.end))
                if note_end == note_start:
                    note_end = min(note_start + 1, len(quaver) - 1) # guitar/bass plunk typically results in a very short note duration. These note should be quantized to 1 instead of 0.
                pr_matrix[note_start, note.pitch, 0] = note_end - note_start
                pr_matrix[note_start, note.pitch, 1] = note.velocity

                #compute quantization error. A song with very high error (e.g., triple-quaver songs) will be discriminated and therefore discarded.
                if note_end == note_start:
                    qt_error.append(np.abs(quaver[note_start] - note.start) / (quaver[note_start] - quaver[note_start-1]))
                else:
                    qt_error.append(np.abs(quaver[note_start] - note.start) / (quaver[note_end] - quaver[note_start]))
                
            control_matrix = np.ones((len(quaver), 128, 1)) * -1
            for control in track.control_changes:
                #if control.time < time_end:
                #    if len(quaver) == 0:
                #        continue
                control_time = np.argmin(np.abs(quaver - control.time))
                control_matrix[control_time, control.number, 0] = control.value

            pr_matrix = np.concatenate((pr_matrix, control_matrix), axis=-1)
            pr_matrices.append(pr_matrix)
            programs.append(track.program)
            quantization_error.append(np.mean(qt_error))

        return np.array(pr_matrices), programs, quantization_error


def pr2grid(pr_mat, max_note_count=16, max_pitch=127, min_pitch=0,
                       pitch_pad_ind=130, dur_pad_ind=2,
                       pitch_sos_ind=128, pitch_eos_ind=129):
    pr_mat3d = np.ones((len(pr_mat), max_note_count, 6), dtype=int) * dur_pad_ind
    pr_mat3d[:, :, 0] = pitch_pad_ind
    pr_mat3d[:, 0, 0] = pitch_sos_ind
    cur_idx = np.ones(len(pr_mat), dtype=int)
    for t, p in zip(*np.where(pr_mat != 0)):
        pr_mat3d[t, cur_idx[t], 0] = p - min_pitch
        binary = np.binary_repr(min(int(pr_mat[t, p]), 32) - 1, width=5)
        pr_mat3d[t, cur_idx[t], 1: 6] = \
            np.fromstring(' '.join(list(binary)), dtype=int, sep=' ')
        if cur_idx[t] == max_note_count-1:
            continue
        cur_idx[t] += 1
    #print(cur_idx)
    pr_mat3d[np.arange(0, len(pr_mat)), cur_idx, 0] = pitch_eos_ind
    return pr_mat3d


def matrix2midi_with_dynamics(matrices, programs, init_tempo=120, time_start=0, ACC=16):
    """
    Reconstruct a multi-track midi from a 3D matrix of shape (Track. Time, 128, 3).
    """
    tracks = []
    for program in programs:
        track_recon = pyd.Instrument(program=int(program), is_drum=False, name=pyd.program_to_instrument_name(int(program)))
        tracks.append(track_recon)

    indices_track, indices_onset, indices_pitch = np.nonzero(matrices[:, :, :, 0])
    alpha = 1 / (ACC // 4) * 60 / init_tempo #timetep between each quntization bin
    for idx in range(len(indices_track)):
        track_id = indices_track[idx]
        onset = indices_onset[idx]
        pitch = indices_pitch[idx]

        start = onset * alpha
        duration = matrices[track_id, onset, pitch, 0] * alpha
        velocity = matrices[track_id, onset, pitch, 1]

        note_recon = pyd.Note(velocity=int(velocity), pitch=int(pitch), start=time_start + start, end=time_start + start + duration)
        tracks[track_id].notes.append(note_recon)

    for idx in range(len(matrices)):
        cc = []
        control_matrix = matrices[idx, :, :, 2]
        for t, n in zip(*np.nonzero(control_matrix >= 0)):
            start = alpha * t
            #if int(n) > 127 or (int(n) < 0):
            #    print(n)
            #if int(control_matrix[t, n]) > 127 or (int(control_matrix[t, n]) < 0):
            #    print(int(control_matrix[t, n]))
            cc.append(pyd.ControlChange(int(n), int(control_matrix[t, n]), start))
        tracks[idx].control_changes = cc
    
    midi_recon = pyd.PrettyMIDI(initial_tempo=init_tempo)
    midi_recon.instruments = tracks
    return midi_recon


def dataitem2midi(tracks, programs, dynamics, dir, program_check, tempo=100):
    tracks = np.concatenate([tracks[..., np.newaxis], dynamics], axis=-1)
    midi_recon = matrix2midi_with_dynamics(tracks, [program_check[prog]  for prog in programs], tempo)
    return midi_recon


def mixture2midi(grid_mix, tempo=100):
    pr = grid2pr(grid_mix.reshape(-1, 32, 6).detach().cpu().numpy(), max_note_count=32)
    recon = matrix2midi(pr[np.newaxis, ...], [0], tempo)
    return recon

def slakh_program_mapping(self, programs):
        return np.array([EMBED_PROGRAM_MAPPING[SLAKH_PROGRAM_MAPPING[program]] for program in programs])

def midi2load(midi_data):

    #midi_data = pyd.PrettyMIDI(midi_file_path)
    #_, tempo = midi_data.get_tempo_changes()

    #all_src_midi = pyd.PrettyMIDI(os.path.join(slakh_split, song, 'all_src.mid'))
    for ts in midi_data.time_signature_changes:
        in_ts = (((ts.numerator == 2) or (ts.numerator == 4)) and (ts.denominator == 4))
        assert in_ts == 1, 'Input time signature not 2/4 or 4/4'

    # Iterate through each instrument (track) in the MIDI file
    track_midi = []
    programs = []
    for i, instrument in enumerate(midi_data.instruments):
        # Create a new PrettyMIDI object
        individual_midi = pyd.PrettyMIDI()
        if instrument.is_drum:
            continue
        # Add the instrument to the new MIDI object
        individual_midi.instruments.append(instrument)
        # Construct the output file name
        output_file_name = f"track_{i + 1}.mid"
        # Save the individual track to a MIDI file
        #individual_midi.write(output_file_name)
        track_midi.append(individual_midi)
        programs.append(instrument.program)
        # Print the program number (instrument number) for the current track
        #print(f"Track {i + 1} - Program Number: {instrument.program}")

    
    ACC = 4 # quantize each beat as 4 positions
    SAMPLE_BAR_LEN = 8
    sample_len = 16 * SAMPLE_BAR_LEN
    #tracks = os.path.join(slakh_split, song, 'MIDI')
    #track_names = os.listdir(tracks)
    #track_midi = [pyd.PrettyMIDI(os.path.join(tracks, track)) for track in track_names]
    #track_meta = yaml.safe_load(open(os.path.join(slakh_split, song, 'metadata.yaml'), 'r'))['stems']

    if len(midi_data.get_beats()) >= max([len(midi.get_beats()) for midi in track_midi]):
        beats = midi_data.get_beats()
        downbeats = midi_data.get_downbeats()
    else:
        beats = track_midi[np.argmax([len(midi.get_beats()) for midi in track_midi])].get_beats()
        downbeats = track_midi[np.argmax([len(midi.get_beats()) for midi in track_midi])].get_downbeats()

    beats = np.append(beats, beats[-1] + (beats[-1] - beats[-2]))
    quantize = interp1d(np.array(range(0, len(beats))) * ACC, beats, kind='linear')
    quaver = quantize(np.array(range(0, (len(beats) - 1) * ACC)))
    
    pr_matrices = []
    #programs = []
    dynamic_matrices = []
    
    break_flag = 0
    for idx, midi in enumerate(track_midi):
        #meta = track_meta[track_names[idx].replace('.mid', '')]
        if midi.instruments[0].is_drum:
            continue    #let's skip drum for now
        else:
            pr_matrix, _, track_qt = midi2matrix_with_dynamics(midi, quaver)
            if track_qt[0] > .2:
                break_flag = 1
                break
            pr_matrices.append(pr_matrix[..., 0])
            dynamic_matrices.append(pr_matrix[..., 1:])
            #programs.append(midi.program)
    #if break_flag:
    #    continue    #skip the pieces with very large quantization error. This pieces are possibly triple-quaver songs
    
    pr_matrices = np.concatenate(pr_matrices, axis=0)
    programs = np.array(programs)
    dynamic_matrices = np.concatenate(dynamic_matrices, axis=0)

    downbeat_indicator = np.array([int(t in downbeats) for t in quaver])
    # np.savez_compressed(os.path.join(save_split, f'{song}.npz'),\
    #                 tracks = pr_matrices,\
    #                 programs = programs,\
    #                 db_indicator = downbeat_indicator,\
    #                 dynamics = dynamic_matrices)
        

    tracks = pr_matrices   #(n_track, time, 128)

    memory = dict({'tracks': [],
                    'programs': [],
                    'dynamics': [],
                    'dir': []
                    })
    #anchor_list = []
    """clipping""" 
    # if (self.mode == 'train') and (self.split =='validation'):
    #     # during model training, no overlapping for validation set
    #     for i in range(0, tracks.shape[1], sample_len):
    #         if i + sample_len >= tracks.shape[1]:
    #             break
    #         self.anchor_list.append((len(self.memory['tracks']), i))  #(song_id, start, total_length)
    # else:
    # otherwise, hop size is 1-bar
    downbeats = np.nonzero(downbeat_indicator)[0]
    # for i in range(0, len(downbeats), 1):
    #     if downbeats[i] + sample_len >= tracks.shape[1]:
    #         break
    #     self.anchor_list.append((len(memory['tracks']), downbeats[i]))  #(song_id, start)
    # #self.anchor_list.append((len(self.memory['tracks']), max(0, (tracks.shape[1]-sample_len))))
    # memory['tracks'].append(tracks)
    # memory['programs'].append(slakh_program_mapping(programs))
    # memory['dir'].append(midi_file_path)

    # memory['dynamics'].append(dynamic_matrices)
    programs = np.array([EMBED_PROGRAM_MAPPING[SLAKH_PROGRAM_MAPPING[program]] for program in programs])#slakh_program_mapping(programs)
    midi_file_path = ''
    tracks = tracks[:,:sample_len]
    dynamic_matrices = dynamic_matrices[:,:sample_len]

    return tracks, programs, dynamic_matrices, midi_file_path