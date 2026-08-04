// Fills in missing lat/lng in data/travels.yaml and data/stays.yaml using
// the Nominatim (OpenStreetMap) geocoding API. Run with: npm run geocode

import { readFileSync, writeFileSync } from 'node:fs';
import { setTimeout as sleep } from 'node:timers/promises';
import { parse } from 'yaml';

const NOMINATIM_URL = 'https://nominatim.openstreetmap.org/search';
const USER_AGENT = 'saashanair-website-travel-map (github.com/saashanair/saashanair-website)';
const REQUEST_DELAY_MS = 1000; // Nominatim usage policy: max 1 req/sec

const DATA_FILES = [
  {
    path: new URL('../data/travels.yaml', import.meta.url),
    header: `# Master list of places visited, used to power the travel map.
#
# Each entry needs \`name\` and \`country\` at minimum. Leave \`lat\`/\`lng\` blank
# for new entries — run \`npm run geocode\` to fill them in automatically.
#
# If a note exists under content/notes/ for a place (categories: travel,
# title matching \`name\`), the map will link to it automatically.
#
# Optional fields:
#   blurb: "..."      — a short note shown in the popup, independent of any linked
#                        note (a place can have a blurb, a linked note, both, or neither)
#   visits: [2019, 2023] — years visited, shown as a tag in the popup. Independent of
#                        blurb — a blurb isn't tied to any one visit in the list.
#
# Places lived in (not just visited) live in data/stays.yaml instead.
`
  },
  {
    path: new URL('../data/stays.yaml', import.meta.url),
    header: `# Places lived in (not just visited), used to power the amber "lived here"
# pins/clusters on the travel map. Each entry needs \`name\` and \`country\` at
# minimum. Leave \`lat\`/\`lng\` blank for new entries — run \`npm run geocode\`
# to fill them in automatically.
#
# If a note exists under content/notes/ for a place (categories: travel,
# title matching \`name\`), the map will link to it automatically.
#
# Optional fields:
#   from: 2021        — year moved there
#   to: 2023           — year left (blank means still there / ongoing)
#   blurb: "..."      — a short note shown in the popup, independent of any linked
#                        note (a place can have a blurb, a linked note, both, or neither)
`
  }
];

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

function serialize(header, places) {
  const entries = places.map((place) => {
    const lines = Object.entries(place).map(([key, value]) => {
      // Blank rather than the literal string "null" for any unset field
      // (lat/lng awaiting geocoding, or an unused optional field).
      if (value === null || value === undefined) return `${key}:`;
      return `${key}: ${serializeValue(value)}`;
    });
    return lines.map((line, i) => (i === 0 ? `- ${line}` : `  ${line}`)).join('\n');
  });
  return `${header}\n${entries.join('\n\n')}\n`;
}

let totalUpdated = 0;

for (const { path, header } of DATA_FILES) {
  const places = parse(readFileSync(path, 'utf8'));

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
    writeFileSync(path, serialize(header, places));
    console.log(`Updated ${updated} place(s) in ${path.pathname.split('/').pop()}\n`);
  }

  totalUpdated += updated;
}

if (totalUpdated === 0) {
  console.log('Nothing to geocode, all places already have coordinates.');
}
