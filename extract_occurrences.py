
from bs4 import BeautifulSoup
import urllib
import http.cookiejar
import re
import time
import sys
import requests
from fake_useragent import UserAgent
from json import dumps


def add_url_params(url, params):
    """ Add GET params to provided URL being aware of existing.

    :param url: string of target URL
    :param params: dict containing requested params to be added
    :return: string with updated URL

    >> url = 'http://stackoverflow.com/test?answers=true'
    >> new_params = {'answers': False, 'data': ['some','values']}
    >> add_url_params(url, new_params)
    'http://stackoverflow.com/test?data=some&data=values&answers=false'
    """
    # Unquoting URL first so we don't loose existing args
    url = urllib.parse.unquote(url)
    # Extracting url info
    parsed_url = urllib.parse.urlparse(url)
    # Extracting URL arguments from parsed URL
    get_args = parsed_url.query
    # Converting URL arguments to dict
    parsed_get_args = dict(urllib.parse.parse_qsl(get_args))
    # Merging URL arguments dict with new params
    parsed_get_args.update(params)

    # Bool and Dict values should be converted to json-friendly values
    # you may throw this part away if you don't like it :)
    parsed_get_args.update(
        {k: dumps(v) for k, v in parsed_get_args.items()
         if isinstance(v, (bool, dict))}
    )

    # Converting URL argument to proper query string
    encoded_get_args = urllib.parse.urlencode(parsed_get_args, doseq=True)
    # Creating new parsed result object based on provided with new
    # URL arguments. Same thing happens inside of urlparse.
    new_url = urllib.parse.ParseResult(
        parsed_url.scheme, parsed_url.netloc, parsed_url.path,
        parsed_url.params, encoded_get_args, parsed_url.fragment
    ).geturl()

    return new_url

def get_num_results(search_term, start_date, end_date):
    """
    Helper method, sends HTTP request and returns response payload.
    """

    # Open website and read html
    user_agent = UserAgent().random
    query_params = {'q': search_term, 'as_ylo': start_date, 'as_yhi': end_date}
    url = 'https://scholar.google.com/scholar?as_vis=1&hl=en&as_sdt=1,5&'
    url = add_url_params(url, query_params)

    # handler = urllib.request.urlopen(urllib.request.Request(url=url, headers={'User-Agent': user_agent}))
    handler = requests.get(url=url, headers={'User-Agent': user_agent})
    # print(handler)
    # print("******************************")
    # print(handler.url)
    # print("******************************")
    # print(handler.text)
    # print("******************************")
    # print(handler.encoding)
    # print("******************************")
    # print(handler.content)
    # exit()
    # html = handler.read()
    # html = handler.text
    html = handler.content

    # Create soup for parsing HTML and extracting the relevant information
    soup = BeautifulSoup(html, 'html.parser')
    div_results = soup.find("div", {"id": "gs_ab_md"})  # find line 'About ts results (y sec)

    if div_results != None:
        res = re.findall(r'(\d+),?(\d+)?,?(\d+)?\s', div_results.text)  # extract number of search results
        if not res:
            num_results = '0'
        else:
            num_results = ''.join(res[0])  # convert string to number

        success = True
    else:
        success = False
        num_results = 0

    return num_results, success


def get_range(search_term, start_date, end_date):

    fp = open("{}.csv".format(search_term), 'w')
    fp.write("year,results\n")
    print("year,results")

    for date in range(start_date, end_date + 1):

        num_results, success = get_num_results(search_term, date, date)
        if not(success):
            print("It seems that you made too many requests to Google Scholar. Please wait a couple of hours and try again.")
            break
        year_results = "{0},{1}".format(date, num_results)
        print(year_results)
        fp.write(year_results + '\n')
        time.sleep(0.8)

    fp.close()


if __name__ == "__main__":
    get_range(search_term='lead-acid', start_date=1950, end_date=2018)

    # if len(sys.argv) < 3:
    #     print("******")
    #     print("Academic word relevance")
    #     print("******")
    #     print("")
    #     print("Usage: python extract_occurrences.py '<search term>' <start date> <end date>")

    # else:
    #     try:
    #         search_term = sys.argv[1]
    #         start_date = int(sys.argv[2])
    #         end_date = int(sys.argv[3])
    #         html = get_range(search_term, start_date, end_date)
    #     finally:
    #         cookies.save()
