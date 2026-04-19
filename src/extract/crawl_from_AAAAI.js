const GQL_URL = "https://pollen.aaaai.org/graphql/public";
const STATION_ID = "5effe609-c645-4620-bc99-a3b34934897c";

async function gql(query, variables = {}, retry = 3) {
  for (let i = 0; i < retry; i++) {
    try {
      const r = await fetch(GQL_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ query, variables }),
      });
      return r.json();
    } catch(e) {
      console.log(`  ⚠️ Error try again ${i+1}/${retry}...`);
      await new Promise(res => setTimeout(res, 2000 * (i+1))); 
    }
  }
  return null;
}

function lastDay(year, month) {
  return new Date(year, month, 0).getDate();
}

async function fetchIds(year, month) {
  const from = `${year}-${String(month).padStart(2,"0")}-01`;
  const to   = `${year}-${String(month).padStart(2,"0")}-${lastDay(year, month)}`;
  const r = await gql(`
    query($f: String) {
      allergenCollectionSets(limit: 100, order: "date", filter: $f) { id date }
    }`, { f: `stationId=="${STATION_ID}" && date>="${from}" && date<="${to}"` });
  return r?.data?.allergenCollectionSets || [];
}

async function fetchDetail(id) {
  const r = await gql(`
    query ($id: ID) {
      allergenCollectionSet (id: $id) {
        id date
        allergenCollections {
          value
          allergen { name category }
        }
      }
    }`, { id });
  return r?.data?.allergenCollectionSet;
}

const CACHE_KEY = "pollen_rows";
let rows = JSON.parse(localStorage.getItem(CACHE_KEY) || "[]");
const done = new Set(rows.map(r => r.date));
console.log(`📦  ${rows.length} `);

const YEARS  = [2022, 2023, 2024, 2025];
const MONTHS = [3, 4, 5, 6, 7, 8, 9, 10];

for (const year of YEARS) {
  for (const month of MONTHS) {
    const days = await fetchIds(year, month);
    const todo = days.filter(d => !done.has(d.date)); 
    console.log(`📅 ${year}/${month}: ${days.length} day | need to: ${todo.length}`);

    for (let i = 0; i < todo.length; i++) {
      const detail = await fetchDetail(todo[i].id);
      if (!detail) continue;

      const row = { date: detail.date, year, month };
      for (const col of detail.allergenCollections || []) {
        const cat = col.allergen.category;
        row[cat] = (row[cat] || 0) + (col.value || 0);
      }
      rows.push(row);
      done.add(detail.date);

      if (i % 10 === 0) {
        localStorage.setItem(CACHE_KEY, JSON.stringify(rows));
        console.log(`  💾 Done saving cache: ${rows.length} days`);
      }
      await new Promise(res => setTimeout(res, 150));
    }
  }
}

localStorage.setItem(CACHE_KEY, JSON.stringify(rows));

const monthly = {};
for (const row of rows) {
  const key = `${row.year}-${String(row.month).padStart(2,"0")}`;
  if (!monthly[key]) monthly[key] = { year: row.year, month: row.month, days: 0, TREE:[], WEED:[], GRASS:[], MOLD:[] };
  monthly[key].days++;
  for (const cat of ["TREE","WEED","GRASS","MOLD"]) {
    if (row[cat] != null) monthly[key][cat].push(row[cat]);
  }
}

const avg = arr => arr.length ? (arr.reduce((a,b)=>a+b,0)/arr.length).toFixed(1) : 0;
const max = arr => arr.length ? Math.max(...arr) : 0;

const headers = ["year","month","n_days","TREE_avg","TREE_max","WEED_avg","WEED_max","GRASS_avg","GRASS_max","MOLD_avg","MOLD_max"];
const csvRows = [headers.join(",")];
for (const key of Object.keys(monthly).sort()) {
  const m = monthly[key];
  csvRows.push([
    m.year, m.month, m.days,
    avg(m.TREE), max(m.TREE),
    avg(m.WEED), max(m.WEED),
    avg(m.GRASS), max(m.GRASS),
    avg(m.MOLD), max(m.MOLD),
  ].join(","));
}

const csv = csvRows.join("\n");
console.log("\n Done!\n" + csv);

const blob = new Blob([csv], { type: "text/csv" });
const a = document.createElement("a");
a.href = window.URL.createObjectURL(blob);
a.download = "pollen_nyc_apr_november_2022_2025.csv";
a.click();

localStorage.removeItem(CACHE_KEY);