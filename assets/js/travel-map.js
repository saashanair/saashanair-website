document.addEventListener('DOMContentLoaded', () => {
  const el = document.getElementById('travel-map');
  if (!el) return;

  const places = JSON.parse(el.dataset.travels);
  const map = L.map(el).setView([20, 0], 2);

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 18
  }).addTo(map);

  const markers = places.map((place) => {
    const hasNote = Boolean(place.url);
    const marker = L.circleMarker([place.lat, place.lng], {
      radius: 8,
      weight: 2,
      color: hasNote ? '#2563eb' : '#94a3b8',
      fillColor: hasNote ? '#3b82f6' : '#cbd5e1',
      fillOpacity: 0.9
    });
    marker.bindPopup(popupHTML(place, hasNote));
    return marker.addTo(map);
  });

  if (markers.length) {
    map.fitBounds(L.featureGroup(markers).getBounds().pad(0.2));
  }
});

function popupHTML(place, hasNote) {
  const title = `<strong>${escapeHTML(place.name)}</strong><br>${escapeHTML(place.country)}`;
  if (!hasNote) return title;

  const images = (place.images || [])
    .map(
      (src) =>
        `<img src="${escapeHTML(src)}" alt="${escapeHTML(place.name)}" style="width:72px;height:72px;object-fit:cover;margin:2px;border-radius:4px;">`
    )
    .join('');

  const summary = place.summary ? `<p style="margin:6px 0;">${escapeHTML(place.summary)}</p>` : '';

  return `${title}${summary}<div style="display:flex;flex-wrap:wrap;max-width:240px;">${images}</div><a href="${escapeHTML(place.url)}">Read full recommendations &rarr;</a>`;
}

function escapeHTML(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
