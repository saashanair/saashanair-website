// Fills in missing lat/lng in data/travels.yaml using the Nominatim
// (OpenStreetMap) geocoding API. Run with: npm run geocode

import { readFileSync, writeFileSync } from 'node:fs';
import { setTimeout as sleep } from 'node:timers/promises';
import { parse } from 'yaml';

const DATA_FILE = new URL('../data/travels.yaml', import.meta.url);
const NOMINATIM_URL = 'https://nominatim.openstreetmap.org/search';
const USER_AGENT = 'saashanair-website-travel-map (github.com/saashanair/saashanair-website)';
const REQUEST_DELAY_MS = 1000; // Nominatim usage policy: max 1 req/sec

const HEADER = `# Master list of places visited, used to power the travel map.
#
# Each entry needs \`name\` and \`country\` at minimum. Leave \`lat\`/\`lng\` blank
# for new entries — run \`npm run geocode\` to fill them in automatically.
#
# If a note exists under content/notes/ for a place (categories: travel,
# title matching \`name\`), the map will link to it automatically.
#
# Optional fields:
#   type: lived       — marks the pin/cluster as a place lived in, not just visited
#   blurb: "..."      — a short note shown in the popup, independent of any linked
#                        note (a place can have a blurb, a linked note, both, or neither)
#   visits: [2019, 2023] — years visited, shown as a tag in the popup. Independent of
#                        blurb — a blurb isn't tied to any one visit in the list.
`;

async function geocode(name, country) {
  const url = new URL(NOMINATIM_URL);
  url.searchParams.set('q', `${name}, ${country}`);
  url.searchParams.set('format', 'json');
  url.searchParams.set('limit', '1');

  const res = await fetch(url, { headers: { 'User-Agent': USER_AGENT } });
  if (!res.ok) throw new Error(`Nominatim request failed: ${res.status}`);

  const [result] = await res.json();
  if (!result) return null;

  return { lat: round(Number(result.lat)), lng: round(Number(result.lon)) };
}

function round(coord) {
  return Math.round(coord * 1e5) / 1e5; // ~1m precision
}

// A JSON string literal is also a valid YAML double-quoted scalar, so this
// safely handles colons, quotes, and other YAML-significant characters.
const quote = (str) => JSON.stringify(str);

function serializeValue(value) {
  if (Array.isArray(value)) {
    return `[${value.map((v) => (typeof v === 'string' ? quote(v) : v)).join(', ')}]`;
  }
  return typeof value === 'string' ? quote(value) : value;
}

function serialize(places) {
  const entries = places.map((place) => {
    const lines = Object.entries(place).map(([key, value]) => {
      if (key === 'lat' || key === 'lng') return `${key}: ${value ?? ''}`;
      return `${key}: ${serializeValue(value)}`;
    });
    return lines.map((line, i) => (i === 0 ? `- ${line}` : `  ${line}`)).join('\n');
  });
  return `${HEADER}\n${entries.join('\n\n')}\n`;
}

const places = parse(readFileSync(DATA_FILE, 'utf8'));

let updated = 0;
for (const place of places) {
  if (place.lat && place.lng) continue;

  process.stdout.write(`Geocoding ${place.name}, ${place.country}... `);
  const coords = await geocode(place.name, place.country);

  if (!coords) {
    console.log('not found, skipping');
    continue;
  }

  place.lat = coords.lat;
  place.lng = coords.lng;
  console.log(`${coords.lat}, ${coords.lng}`);
  updated++;

  await sleep(REQUEST_DELAY_MS);
}

if (updated > 0) {
  writeFileSync(DATA_FILE, serialize(places));
  console.log(`\nUpdated ${updated} place(s) in data/travels.yaml`);
} else {
  console.log('\nNothing to geocode, all places already have coordinates.');
}
