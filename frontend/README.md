# Frontend - 0DTE Options Flow Monitor

Angular 21 application with real-time WebSocket data visualization using Signals architecture.

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm start

# Access at http://localhost:4200
```

## Architecture

**Framework:** Angular 21 (standalone components)  
**Styling:** Tailwind CSS v4  
**State Management:** Angular Signals  
**Real-time:** Native WebSocket API

## Project Structure

```
src/
├── app/
│   ├── app.component.ts           # Root component with header
│   ├── app.config.ts              # App configuration
│   ├── data-stream.service.ts     # WebSocket service
│   └── strike-grid/               # Real-time strike grid
│       ├── strike-grid.component.ts
│       ├── strike-grid.component.html
│       └── strike-grid.component.css
├── main.ts                        # Bootstrap
├── styles.css                     # Global Tailwind imports
└── index.html
```

## Key Components

### `DataStreamService`
WebSocket client that connects to backend and manages flow data:

```typescript
// Connects automatically on initialization
constructor() { this.connect(); }

// Exposes reactive signal
public flowData: WritableSignal<FlowMap> = signal({});

// Auto-reconnects on disconnect
```

**Connection:** `ws://localhost:8000/ws/stream`

### `StrikeGridComponent`
Real-time grid display:
- Subscribes to `flowData` signal
- Updates automatically on backend broadcasts
- Displays: ticker, volume, premium, delta flow, last price

### `AppComponent`
Root layout with header and grid container.

## Data Flow

```
Backend WebSocket (250ms)
         │
         ▼
   DataStreamService
         │
         ▼
   flowData Signal
         │
         ▼
  StrikeGridComponent
         │
         ▼
      DOM Update
```

## Development

### Run dev server
```bash
npm start
# Opens http://localhost:4200
# Hot reload enabled
```

### Build for production
```bash
npm run build
# Output: dist/
```

### Generate component
```bash
ng generate component component-name
```

## Tailwind CSS v4

**Configuration:** Automatic - no config file needed

**Import:** Via `@import "tailwindcss"` in `styles.css`

**Usage:**
```html
<div class="bg-gray-900 text-white p-4">
  <h1 class="text-xl font-bold">Title</h1>
</div>
```

## WebSocket Protocol

### Connection
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/stream');
```

### Message Format
```typescript
interface FlowMap {
  [ticker: string]: FlowMetrics;
}

interface FlowMetrics {
  cumulative_volume: number;
  cumulative_premium: number;
  last_price: number;
  net_delta_flow: number;
  net_gamma_flow: number;
  delta: number;
  gamma: number;
  strike_price: number;
  type: string;  // 'C' or 'P'
  expiration: string;
}
```

## Debugging

### Check WebSocket connection
1. Open browser DevTools
2. Network tab → WS filter
3. Look for `ws://localhost:8000/ws/stream`
4. Status should be `101 Switching Protocols`

### Console messages
```javascript
// Expected logs:
🔌 Connecting to 0DTE Stream...
✅ Connected to Stream
```

### Common Issues

**Connection refused:**
- Ensure backend is running on port 8000
- Check `uv run fastapi dev src/main.py` is active

**No data displayed:**
- Check browser console for errors
- Verify WebSocket messages in Network tab
- Check backend logs for "Loaded X trades"

**Styles not applying:**
- Clear `.angular/cache`
- Restart dev server
- Check `styles.css` has `@import "tailwindcss"`

## Testing

```bash
# Run tests
npm test

# Uses Vitest
```

## Build

```bash
# Production build
npm run build

# Output optimized for performance
# Files in dist/ directory
```

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

## Performance

**Initial Load:** < 1s  
**Update Latency:** < 250ms (backend throttle)  
**Memory:** ~50MB typical  
**Bundle Size:** ~200KB gzipped

## Code Style

**Formatting:** Automatic via Prettier (see `package.json`)

**Conventions:**
- Standalone components (no NgModules)
- Signals for reactive state
- TypeScript strict mode
- Inline templates for small components

## Dependencies

Key packages:
- `@angular/core` - Framework
- `@angular/common` - Common directives
- `rxjs` - Reactive programming
- `tailwindcss` - Styling

Dev dependencies:
- `@angular/cli` - Tooling
- `vitest` - Testing
- `typescript` - Language

## IDE Setup

**Recommended:** VS Code with Angular Language Service extension

**Settings:**
```json
{
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "esbenp.prettier-vscode"
}
```

## Deployment

```bash
# Build
npm run build

# Serve from dist/ directory
cd dist/frontend/browser
python -m http.server 8080

# Or use any static file server
```

**Note:** Ensure backend WebSocket URL is configured for production environment.
