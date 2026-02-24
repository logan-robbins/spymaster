import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    port: 5174,
  },
  build: {
    rollupOptions: {
      input: {
        'vp-stream': 'vp-stream.html',
        'experiments': 'experiments.html',
      },
    },
  },
});
