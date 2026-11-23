import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig, loadEnv } from 'vite';

export default defineConfig(({ mode }) => {
  // Backend the dev server proxies to. Should match web.port in config.yaml;
  // override with DNDSCRIBE_BACKEND=http://localhost:3002 (env or .env file).
  const env = loadEnv(mode, '.', 'DNDSCRIBE_');
  const backend = env.DNDSCRIBE_BACKEND || 'http://localhost:3001';

  return {
    plugins: [sveltekit()],
    server: {
      proxy: {
        // ws: /api/dj/listen (test channel audio) is a WebSocket.
        '/api': { target: backend, ws: true },
        '/recordings': backend,
        '/ws': {
          target: backend.replace(/^http/, 'ws'),
          ws: true,
        },
      },
    },
  };
});
