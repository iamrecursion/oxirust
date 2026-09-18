// Node smoke tests for the OxiEphemeris WebAssembly bindings.
//
// These exercise every exported function against the real DE440 ephemeris
// and assert the results match the values the CLI and the Python binding
// produce for the same input — the shared facade guarantees they are
// bit-for-bit identical across all three surfaces.
//
// The reference chart is a synthetic, non-personal fixture: the Unix epoch
// (1970-01-01T00:00:00Z) at the Royal Observatory, Greenwich, with a
// companion chart at Y2K noon in Paris.
//
// Build the module first, then run the suite:
//
//   wasm-pack build crates/oxiephemeris-wasm --release --target nodejs \
//     --out-dir tests/node/pkg
//   node --test crates/oxiephemeris-wasm/tests/node
//
// The suite skips cleanly if the built module or the DE440 file is absent.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const pkgEntry = join(here, 'pkg', 'oxiephemeris_wasm.js');
const dePath = join(here, '..', '..', '..', '..', 'data', 'de440', 'linux_p1550p2650.440');

const EPOCH = { date: '1970-01-01T00:00:00Z', lat: 51.4779, lon: 0.0 };
const Y2K = { date: '2000-01-01T12:00:00Z', lat: 48.8566, lon: 2.3522 };

const missing = !existsSync(pkgEntry)
  ? `built module not found at ${pkgEntry} — run \`npm run build\` in this directory first`
  : !existsSync(dePath)
    ? `DE440 fixture not found at ${dePath}`
    : null;

const oxi = missing ? null : await import(pkgEntry);
const de = missing ? null : new Uint8Array(readFileSync(dePath));

// node:test treats the mere presence of a `skip` key as "skip" (even when
// its value is null), so omit the key entirely unless we mean to skip.
const opts = missing ? { skip: missing } : {};

const person = ({ date, lat, lon }) => ({ date, lat, lon });

test('natal chart matches the reference values', opts, () => {
  const chart = JSON.parse(oxi.natal_json(de, JSON.stringify(EPOCH)));
  assert.equal(chart.kind, 'natal');
  assert.equal(chart.sect, 'nocturnal');

  const sun = chart.bodies.find((b) => b.body === 'Sun');
  assert.equal(sun.sign, 'Capricorn');
  assert.ok(Math.abs(sun.sign_degrees - 10.156) < 0.01, `Sun deg ${sun.sign_degrees}`);
  assert.equal(sun.house, 4);

  const mars = chart.dignities.find((d) => d.body === 'Mars');
  assert.equal(mars.score, 3);

  const dist = chart.distribution;
  assert.equal(dist.fire + dist.earth + dist.air + dist.water, 10);
});

test('natal Turtle carries the expected facts', opts, () => {
  const ttl = oxi.natal_turtle(de, JSON.stringify(EPOCH), undefined);
  assert.match(ttl, /@prefix oxa:/);
  assert.match(ttl, /oxa:inSign sign:Capricorn/);
  assert.match(ttl, /oxa:dignityScore 3/);
});

test('synastry and composite serialize', opts, () => {
  const pair = JSON.stringify({ a: person(EPOCH), b: person(Y2K) });

  const syn = JSON.parse(oxi.synastry_json(de, pair));
  assert.equal(syn.kind, 'synastry');
  assert.equal(syn.chart_a.length, 12);
  assert.ok(syn.cross_aspects.length > 0);
  assert.match(oxi.synastry_turtle(de, pair, undefined), /oxa:SynastryComparison/);

  const comp = JSON.parse(oxi.composite_json(de, pair));
  assert.equal(comp.composite.length, 12);
  assert.equal(comp.source_charts.length, 2);
  const ttl = oxi.composite_turtle(de, pair, undefined);
  assert.match(ttl, /oxa:CompositeChart/);
  assert.match(ttl, /wasDerivedFrom/);
});

test('transit and progression compute', opts, () => {
  const tr = JSON.parse(
    oxi.transit_json(de, JSON.stringify({ natal: person(EPOCH), transit: '2026-07-10T00:00:00' })),
  );
  assert.equal(tr.kind, 'transit');
  assert.ok(tr.cross_aspects.length > 0);

  const pr = JSON.parse(
    oxi.progress_json(de, JSON.stringify({ natal: person(EPOCH), target: '2010-01-01T00:00:00' })),
  );
  assert.equal(pr.kind, 'progression');
  assert.ok(pr.elapsed_years > 35.0);
});

test('a malformed request throws', opts, () => {
  assert.throws(() =>
    oxi.natal_json(de, JSON.stringify({ ...EPOCH, system: 'martian' })),
  );
});
