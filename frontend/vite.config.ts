import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    port: 5174,
  },
  build: {
    rollupOptions: {
      input: {
        'stream': 'stream.html',
        'experiments': 'experiments.html',
        'jobs': 'jobs.html',
        'model_studio': 'model_studio.html',
        'serving_registry': 'serving_registry.html',
      },
    },
  },
});
