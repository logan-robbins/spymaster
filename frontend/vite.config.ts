import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    port: 5174,
  },
  build: {
    rollupOptions: {
      input: {
        'vacuum-pressure': 'vacuum-pressure.html',
        'experiments': 'experiments.html',
      },
    },
  },
});
