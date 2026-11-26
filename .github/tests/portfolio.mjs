import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import {chromium} from "playwright-core";

const base = process.env.SITE_URL;
const browser = await chromium.launch({channel:"chrome"});
try {
  const context = await browser.newContext();
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror",e => errors.push(e.message));
  page.setDefaultTimeout(12000);
  const response = await page.goto(base);
  assert.equal(response.status(),200);
  assert.equal(await page.title(),"터틀 포지션 계산기");
  await page.getByRole("button",{name:"예시 불러오기"}).click();
  await page.waitForFunction(() => document.querySelector("#stop").textContent === "205,000원");

  // Use synthetic quotes and holdings, so no personal trades enter CI logs.
  const codes = ["100001","100002","100003","100004"];
  const rows = [];
  for (let day = new Date("2025-11-01T12:00Z");day <= new Date("2026-01-16T12:00Z");day.setUTCDate(day.getUTCDate()+1)) {
    if (![0,6].includes(day.getUTCDay())) rows.push([day.toISOString().slice(0,10),10000,10100,9900]);
  }
  await context.route("https://aikstockdata.com/**",async route => {
    const code = route.request().url().match(/(\d{6})_history/)?.[1];
    if (!code) return route.abort();
    await route.fulfill({json:{code,name_ko:"테스트 "+code,columns:["date","close","high","low"],rows}});
  });
  async function hasText(id,value) {
    await page.waitForFunction(([id,value]) => document.getElementById(id).textContent === value,[id,value]);
  }
  const holdings = page.locator("#bookRows tr");
  const capital = page.locator("#capital");
  await capital.fill("6000000");
  await page.locator("#bulkToggle").click();
  await page.locator("#bulkDate").fill("2026-01-15");
  await page.locator("#bulkText").fill(codes.map((code,i) => [code,10000+i*100,3+i].join(",")).join("\n"));
  await page.locator("#previewBulk").click();
  await hasText("importTitle","4종목 가져오기");
  await page.locator("#confirmImport").click();
  await hasText("holdingCount","4종목");
  await hasText("bookBudget","1,500,000원");
  await page.waitForFunction(() => document.querySelector("#bookRows td[data-label='최초 손절가']").textContent === "9,600원");
  assert.equal(await holdings.count(),4);
  assert.equal(await holdings.first().locator("[data-label='첫 추가매수가']").innerText(),"10,100원");
  assert.equal(await page.locator("#count").inputValue(),"4");
  await page.reload();
  await hasText("holdingCount","4종목");
  assert.equal(await capital.inputValue(),"6000000");
  assert.equal(await holdings.first().locator("[data-label='보유 수량']").innerText(),"3주");
  await capital.fill("12000000");
  await hasText("bookBudget","3,000,000원");
  await holdings.first().locator(".position-name").click();
  await page.locator("#held").fill("10");
  await page.locator("#saveHolding").click();
  await page.waitForFunction(() => document.querySelector("#bookRows td[data-label='보유 수량']").textContent === "10주");
  assert.equal(await holdings.count(),4);
  const downloadPromise = page.waitForEvent("download");
  await page.locator("#exportBook").click();
  const backup = await readFile(await (await downloadPromise).path());
  assert.equal(JSON.parse(backup).positions.length,4);
  page.once("dialog",d => d.accept());
  await holdings.last().getByRole("button",{name:/삭제/}).click();
  await hasText("holdingCount","3종목");
  await hasText("bookBudget","4,000,000원");
  await page.locator("#bookFile").setInputFiles({name:"backup.json",mimeType:"application/json",buffer:backup});
  await page.locator("#confirmImport").click();
  await hasText("holdingCount","4종목");
  assert.equal(await holdings.count(),4);
  assert.equal(await capital.inputValue(),"12000000");
  // A fresh browser can import a private link without sending its fragment to the server.
  const fresh = await browser.newContext({viewport:{width:390,height:844}});
  const mobile = await fresh.newPage();
  await mobile.route("https://aikstockdata.com/**",route => route.abort());
  await mobile.goto(base.split("#")[0]+"#holdings=2026-11-26;750000;068270:150000");
  await mobile.locator("#confirmImport").click();
  await mobile.waitForFunction(() => document.querySelector("#bookRows tr") && !location.hash);
  await mobile.reload();
  assert.equal(await mobile.locator("#capital").inputValue(),"750000");
  assert.equal(await mobile.locator("#bookRows tr").count(),1);
  assert.equal(await mobile.locator("#bookRows td[data-label='보유 수량']").textContent(),"미입력");
  assert.equal(await mobile.evaluate(() => document.documentElement.scrollWidth <= innerWidth),true);
  assert.deepEqual(errors,[]);
  console.log("Verified four holdings, shared budget, recalculation, persistence, edit, deletion, backup, import and mobile view.");
} finally {
  await browser.close();
}
