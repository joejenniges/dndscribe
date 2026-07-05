// Svelte 5 runes-mode shared state for the audio player.
// Using .svelte.ts extension so $state works at module level.

let open = $state(false);
let filenames = $state<string[]>([]);
let speaker = $state('');
let speakerColor = $state('');

export const audioPlayer = {
  get open() { return open; },
  set open(v: boolean) { open = v; },
  get filenames() { return filenames; },
  get speaker() { return speaker; },
  get speakerColor() { return speakerColor; },
};

export function openAudioPlayer(fnames: string[], spk: string, color: string) {
  filenames = fnames;
  speaker = spk;
  speakerColor = color;
  open = true;
}

export function closeAudioPlayer() {
  open = false;
}
