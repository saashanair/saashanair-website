# Travel map snagging list

Ideas to work through for the `/travel-map` page, roughly in the order raised.

- [ ] **Multi-place trips sharing one note** — e.g. Bangkok, Khao Sok, Phuket were one
      Thailand trip with only enough content for a single note, not three. Title-matching
      (note title == place name) can't cover this. Add an optional `note:` field to a
      travel entry that explicitly points at a note slug, overriding the automatic
      title match. Default stays zero-config for the common single-place case.

- [ ] **Pins are too big / cluttered** — shrink the pin SVG size (currently 30x40).

- [ ] **Hover tooltip with place name** — use Leaflet's native `bindTooltip` so the
      name shows on hover, reducing the need to click every pin to identify it.

- [ ] **Declutter pins that are close together** — use the `Leaflet.markercluster`
      plugin (pins within a radius collapse into a count badge, split apart on zoom)
      rather than fixed country/region pins. Clustering adapts continuously with zoom
      and keeps exact geography; region pins would need every place manually assigned
      to a region and lose precision. Tradeoff: one more CDN dependency, same pattern
      as Leaflet itself.

- [ ] **Photos without a full written note** — folder convention: drop photos into
      `static/images/travels/<slugified-name>/`, template auto-discovers them at
      build time (Hugo's `os.ReadDir`). No YAML editing, no note required.

- [ ] **Indicate "lived here" vs. "visited"** — optional `type: lived` field on an
      entry (default `visited`), its own pin color/icon and popup phrasing. Could pair
      with optional `from`/`to` year fields.

- [ ] **Short blurbs without a full recommendations post** — optional `blurb:`
      free-text field directly in the yaml entry. Gives a third tier: bare pin → pin +
      short text → pin + full popup with photos and a link to a note.

- [ ] **Track multiple visits to the same place** — optional `visits: [2019, 2023]`
      field (list of years, not just a count), so the popup can say "Visited in 2019
      and 2023" and it's usable later for filtering/sorting.
