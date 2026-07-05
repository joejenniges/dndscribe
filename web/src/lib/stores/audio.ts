import { writable } from 'svelte/store';

export const audioPlayerOpen = writable(false);
export const audioFilenames = writable<string[]>([]);
export const audioSpeaker = writable('');
export const audioSpeakerColor = writable('');

export function openAudioPlayer(filenames: string[], speaker: string, color: string) {
  audioFilenames.set(filenames);
  audioSpeaker.set(speaker);
  audioSpeakerColor.set(color);
  audioPlayerOpen.set(true);
}

export function closeAudioPlayer() {
  audioPlayerOpen.set(false);
}
