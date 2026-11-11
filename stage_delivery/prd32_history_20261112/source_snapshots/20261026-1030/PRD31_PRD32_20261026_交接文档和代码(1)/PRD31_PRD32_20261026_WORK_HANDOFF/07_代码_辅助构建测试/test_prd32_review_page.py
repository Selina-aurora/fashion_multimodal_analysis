from playwright.sync_api import sync_playwright
from pathlib import Path
p=Path('/mnt/data/PRD32_W7_Dev7_人工审核工作台_离线版.html')
with sync_playwright() as w:
    b=w.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox'])
    page=b.new_page(viewport={'width':1366,'height':900},accept_downloads=True)
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.set_content(p.read_text(encoding="utf-8"),wait_until="load")
    print('title',page.title())
    print('nav count',page.locator('.nav-case').count(),'errors',errors)
    assert page.locator('.nav-case').count()==7
    assert page.locator('#compare tbody').count()==0  # actual compare is tbody
    assert page.locator('#compare tr').count()==7
    assert page.locator('#imgLeft').evaluate('(x)=>x.complete && x.naturalWidth > 0')
    # click candidate action: remain PENDING
    page.locator('#applySuggestions').click()
    print('suggest category',page.locator('#category').input_value(),'status',page.locator('#categoryStatus').input_value())
    assert page.locator('#categoryStatus').input_value()=='PENDING'
    # set statuses and gt, ensure verified accepted
    page.locator('#category').select_option('outerwear')
    page.locator('#color').fill('粉红色')
    page.locator('#pattern').fill('纯色')
    page.locator('#design').fill('长袖；前襟扣饰')
    for f in ['categoryStatus','colorStatus','patternStatus','designStatus']:
        page.locator('#'+f).select_option('CONFIRMED')
    page.locator('#bbox').select_option('TARGET_CORRECT')
    page.locator('#reviewStatus').select_option('VERIFIED')
    print('GT eligible',page.locator('#countEligible').inner_text(),'verified',page.locator('#countVerified').inner_text())
    assert page.locator('#countEligible').inner_text()=='1'
    with page.expect_download() as download_info:
       page.locator('#export14').click()
    d=download_info.value
    output=Path('/mnt/data/_test_downloaded.csv');d.save_as(output)
    print('export',d.suggested_filename,'bytes',output.stat().st_size)
    assert 'VERIFIED' in output.read_text('utf-8-sig')
    page.locator('#nextCase').click()
    print('next',page.locator('#caseTitle').inner_text())
    assert 'dev_04' in page.locator('#caseTitle').inner_text()
    page.screenshot(path='/mnt/data/PRD32_review_html_preview.png',full_page=False)
    print('console errors',errors)
    assert not errors
    b.close()
