# HW08_406235002
from selenium import webdriver
import time
from selenium.webdriver.support.select import Select
import pandas as pd

# ----------------------------------------------------------  Web Scraping by Selenium
chromePath = 'chromedriver.exe'
url = 'https://erdb.epa.gov.tw/DataRepository/EnvMonitor/AirPSIValues.aspx?topic1=%E5%A4%A7%E6%B0%' \
      'A3&topic2=%E7%92%B0%E5%A2%83%E5%8F%8A%E7%94%9F%E6%85%8B%E7%9B%A3%E6%B8%AC&subject=%E7%A9%BA%E6%B0%A3%E5%93%81%E8%B3%AA'

driver = webdriver.Chrome(chromePath)
driver.get(url)
time.sleep(3)
driver.maximize_window()
driver.find_element_by_id('ctl00_ContentPlaceHolder1_ucSearchCondition_rblCondition_1').click()  # method of search

driver.find_element_by_id('ctl00_ContentPlaceHolder1_ucSearchCondition_ddlCounty').click()  # County
driver.find_element_by_css_selector("option[value='10009   ']").click()
time.sleep(5)

driver.find_element_by_id('ctl00_ContentPlaceHolder1_ucSearchCondition_ddlSite').click()  # Station
driver.find_element_by_css_selector("option[value='205                 ']").click()
time.sleep(2)

Select(driver.find_element_by_id('ctl00_ContentPlaceHolder1_ucSearchCondition_ddlYear')).select_by_value('2019')
Select(driver.find_element_by_id('ctl00_ContentPlaceHolder1_ucSearchCondition_ddlMonth')).select_by_value('01')
time.sleep(1)  # 至
Select(driver.find_element_by_id('ctl00_ContentPlaceHolder1_ucSearchCondition_ddlYearE')).select_by_value('2019')
Select(driver.find_element_by_id('ctl00_ContentPlaceHolder1_ucSearchCondition_ddlMonthE')).select_by_value('12')
time.sleep(3)
Select(driver.find_element_by_name('ctl00$ContentPlaceHolder1$ucSearchCondition$ddlDay')).select_by_value('01')
Select(driver.find_element_by_name('ctl00$ContentPlaceHolder1$ucSearchCondition$ddlDayE')).select_by_value('30')
time.sleep(1)

driver.find_element_by_id('ctl00_ContentPlaceHolder1_imgSearch').click()  # search
time.sleep(6)

# ---------------------------------------------------------- Data load
# get the pages
web_pages = driver.find_element_by_id('ctl00_ContentPlaceHolder1_MessageBar_lblPage').get_attribute('innerHTML')
n = []
for x in web_pages:
    try:
        int(x)
        n.append(x)
    except Exception as err:
        print(err)

s = "".join(n)
pages = int(s[1:])

# Start load all data
Datasets = []
i = 1
while i <= pages:
    try:
        driver.find_element_by_link_text(str(i)).click()
        trlist = driver.find_elements_by_id('ctl00_ContentPlaceHolder1_ucShareAndExport_gvPrint')[0].find_elements_by_tag_name('tr')
        for y in trlist[1:]:
            dataset = [td.get_attribute("innerHTML") for td in y.find_elements_by_tag_name('td')]
            Datasets.append(dataset)
        i += 1
    except Exception as err:
        print(err)
        if '{"method":"link text","selector":"1"}' in str(err):  # page1 cannot click
            trlist = driver.find_elements_by_id('ctl00_ContentPlaceHolder1_ucShareAndExport_gvPrint')[0].find_elements_by_tag_name('tr')
            for y in trlist[1:]:
                dataset = [td.get_attribute("innerHTML") for td in y.find_elements_by_tag_name('td')]
                Datasets.append(dataset)
            i += 1
            print('Solved the ', err)
        else:
            driver.find_element_by_css_selector("a[href*='Page$%s']" % i).click()  # click '...' for get next pages
            trlist = driver.find_elements_by_id('ctl00_ContentPlaceHolder1_ucShareAndExport_gvPrint')[0].find_elements_by_tag_name('tr')
            for y in trlist[1:]:
                dataset = [td.get_attribute("innerHTML") for td in y.find_elements_by_tag_name('td')]
                Datasets.append(dataset)
            i += 1
            print('Solved the ', err)

trlist = driver.find_elements_by_id('ctl00_ContentPlaceHolder1_ucShareAndExport_gvPrint')[0].find_elements_by_tag_name('tr')
headList = [th.get_attribute("innerHTML") for th in trlist[0].find_elements_by_tag_name('th')]  # innerHTML : the text

# driver.quit()

df = pd.DataFrame(Datasets, columns = headList)

# write file
df.to_excel('data.xlsx')