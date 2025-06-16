import sys, json, time, re, os

from selenium import webdriver

urls = None
url = None
start = 1
if (len(sys.argv) > 1 and not sys.argv[1].isnumeric()):
    _, url = sys.argv
    urls = [ url ]
else:
    with open('./urls.downloaded.alexa.txt') as f:
        urls = [ line.rstrip() for line in f ]
        companies = [ url.split('.')[0] for url in urls ]
    to_remove = []
    for index, company in enumerate(companies):
        if (company in companies[index+1:]):
            to_remove.append(companies[index+1:].index(company) + index + 1)
    print('Removed considering company duplicate in alexa list: %s' % ([ urls[i] for i in to_remove ]))
    urls = [ url for i, url in enumerate(urls) if i not in to_remove ]

    if len(sys.argv) > 1 and sys.argv[1].isnumeric():
        start = int(sys.argv[1])

DOM_collector = '''
let all = [];
function count_words (el) {
    const childs = el.querySelectorAll('*');
    let childs_count = 0,
        word_count = 0;
    for (let i = 0; i < childs.length; i++) {
        const target = childs[i];
        if (target.children.length === 0) {
            childs_count++;
            word_count += target.textContent.split(' ').length;
        }
    }
    if (childs_count === 0) return 0;
    return word_count / childs_count;
}
function offset (el) {
    const rect = el.getBoundingClientRect(),
          win = el.ownerDocument.defaultView;
    return {
        top: rect.top + win.pageYOffset,
        left: rect.left + win.pageXOffset
    };
}
function dimension (el) {
    return {
        height: (el.offsetHeight ? el.offsetHeight : 0),
        width: (el.offsetWidth ? el.offsetWidth : 0)
    };
}
function landmark_parent (el) {
    if (!el)
        return false;
    const tagname = el.tagName.toLowerCase();
    if ((tagname === 'footer' || tagname === 'aside' || tagname === 'main' ||
        tagname === 'form' || tagname === 'header') && label_for(el))
        return tagname;
    if (tagname === 'nav' || tagname === 'section') {
        return tagname;
    }
    const role = el.getAttribute('role');
    if (role) {
        return role;
    }
    return landmark_parent (el.parentElement);
}
function label_for (el) {
    const labelledby = el.getAttribute('aria-labelledby'),
          label = el.getAttribute('aria-label'),
          title = el.getAttribute('title');

    if (labelledby || label || title) {
        return `${el.className} ${labelledby} ${label} ${title}`;
    }
    return false;
}
function get_xpath (target) {
    var xpath = '', tagName, parent = target.parentElement,
        index, children;
    while (parent != null) {
        tagName = target.tagName.toLowerCase();
        children = [].slice.call(parent.children);
        index = children.indexOf(target) + 1;
        xpath = '/' + tagName + '[' + index + ']' + xpath;
        target = parent;
        parent = target.parentElement;
    };
    return xpath;
}
function calculate_weighted_avg (el, attr_call, weight) {
   let childs = Array.from(el.children),
       weighted_sum = 0,
       size = childs.length;

    childs.forEach((child) => {
        const result = calculate_weighted_avg(child, attr_call, weight / 2);
        weighted_sum += attr_call(child) * weight + result.weighted_sum;
        size += result.size;
    });

    return { weighted_sum, size, weighted_avg: weighted_sum / size };
}
return Array.from(document.body.querySelectorAll('*')).map((el) => {
    const position = offset(el);
    const tags = [
        'a', 'abbr', 'acronym', 'address', 'applet', 'area',
        'article', 'aside', 'audio', 'b', 'base', 'basefont',
        'bdi', 'bdo', 'big', 'blockquote', 'body', 'br', 'button',
        'canvas', 'caption', 'center', 'cite', 'code', 'col',
        'colgroup', 'data', 'datalist', 'dd', 'del', 'details',
        'dfn', 'dialog', 'dir', 'div', 'dl', 'dt', 'em', 'embed',
        'fieldset', 'figcaption', 'figure', 'font', 'footer',
        'form', 'frame', 'frameset',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'head', 'header',
        'hr', 'html', 'i', 'iframe', 'img', 'input', 'ins', 'kbd',
        'label', 'legend', 'li', 'link', 'main', 'map', 'mark',
        'meta', 'meter', 'nav', 'noframes', 'noscript', 'object',
        'ol', 'optgroup', 'option', 'output', 'p', 'param',
        'picture', 'pre', 'progress', 'q', 'rp', 'rt', 'ruby',
        's', 'samp', 'script', 'section', 'select', 'small',
        'source', 'span', 'strike', 'strong', 'style', 'sub',
        'summary', 'sup', 'svg', 'table', 'tbody', 'td', 'template',
        'textarea', 'tfoot', 'th', 'thead', 'time', 'title', 'tr',
        'track', 'tt', 'u', 'ul', 'var', 'video', 'wbr'];

    let counter = {};
    tags.forEach((tag) => {
        counter[`${tag}_count`] = el.querySelectorAll(tag).length;
    });

    let averages = {};
    if (el.children.length > 0) {
        averages.avg_top = Array.from(el.children).reduce((acc, child) =>
                acc + (offset(child).top - position.top), 0) / el.children.length;
        averages.weighted_top = calculate_weighted_avg(el, (elem) =>
                offset(elem).top - position.top, 1).weighted_avg;
        averages.sd_top = Math.sqrt(Array.from(el.children).reduce((acc, child) =>
                acc + ((averages.avg_top - offset(child).top)**2), 0) / el.children.length);

        averages.avg_left = Array.from(el.children).reduce((acc, child) =>
                acc + (offset(child).left - position.left), 0) / el.children.length;
        averages.weighted_left = calculate_weighted_avg(el, (elem) =>
                offset(elem).left - position.left, 1).weighted_avg;
        averages.sd_left = Math.sqrt(Array.from(el.children).reduce((acc, child) =>
                acc + ((averages.avg_left - offset(child).left)**2), 0) / el.children.length);

        averages.avg_height = Array.from(el.children).reduce((acc, child) =>
                acc + (dimension(child).height - dimension(el).height), 0) / el.children.length;
        averages.weighted_height = calculate_weighted_avg(el, (elem) =>
                dimension(elem).height, 1).weighted_avg;
        averages.sd_height = Math.sqrt(Array.from(el.children).reduce((acc, child) =>
                acc + ((averages.avg_height - dimension(child).height)**2), 0) / el.children.length);

        averages.avg_width = Array.from(el.children).reduce((acc, child) =>
                acc + (dimension(child).width - dimension(el).width), 0) / el.children.length;
        averages.weighted_width = calculate_weighted_avg(el, (elem) =>
                dimension(elem).width, 1).weighted_avg;
        averages.sd_width = Math.sqrt(Array.from(el.children).reduce((acc, child) =>
                acc + ((averages.avg_width - dimension(child).width)**2), 0) / el.children.length);
    } else {
        averages.avg_top = -1;
        averages.weighted_top = -1;
        averages.sd_top = -1;
        averages.avg_left = -1;
        averages.weighted_left = -1;
        averages.sd_left = -1;
        averages.avg_height = -1;
        averages.weighted_height = -1;
        averages.sd_height = -1;
        averages.avg_width = -1;
        averages.weighted_width = -1;
        averages.sd_width = -1;
    }

    let body = document.body, html = document.documentElement;
    return {
        url: window.location.href,
        tagName: el.tagName,
        role: el.getAttribute('role'),
        top: position.top,
        left: position.left,
        height: dimension(el).height,
        width: dimension(el).width,
        childs_count: el.querySelectorAll('*').length,
        window_height: Math.max(body.scrollHeight, body.offsetHeight, html.clientHeight, html.scrollHeight, html.offsetHeight),
        window_elements_count: document.querySelectorAll('*').length,
        className: el.className,
        parent_landmark: landmark_parent(el.parentElement),
        label: label_for(el),
        xpath: get_xpath(el),
        word_count: count_words(el),
        ...counter,
        ...averages
    };
});
'''
output_dir = './2-output-urls-data'
os.makedirs(output_dir, exist_ok=True)

for i, url in enumerate(urls, start=1):
    if i < start:
        print('already visited %d - http://www.%s' % (i, url))
    else:
        options = webdriver.ChromeOptions()
        options.headless = True
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        # options.add_experimental_option( "prefs",{'profile.managed_default_content_settings.javascript': 2})
        driver = webdriver.Remote(
            command_executor='http://localhost:4444/wd/hub',
            options=options)

        print('visiting %d - http://www.%s' % (i, url))
        count = 0
        while count < 5:
            try:
                driver.set_page_load_timeout(60)
                driver.implicitly_wait(30)
                driver.set_script_timeout(30)
                driver.get('http://%s' % (url))

                time.sleep(10)
                height = driver.execute_script(
                    ''' var body = document.body, html = document.documentElement;
                        return Math.max(body.scrollHeight, body.offsetHeight,
                                        html.clientHeight, html.scrollHeight,
                                        html.offsetHeight)''')
                driver.set_window_size(1080, height)
                time.sleep(10)
                driver.execute_script('''for (var i = 0; i < 1000000; i++) {
                                            clearTimeout(i);
                                            clearInterval(i);
                                         }
                                         window.stop(); ''')
                time.sleep(10)

                match = re.search(r"/(\d+)/", url)
                if match:
                    downloader_folder = match.groups()
                else:
                    domain_id = url.split('.')[0].replace('/', '_')
                    downloader_folder = (domain_id,)
                el_data = driver.execute_script(DOM_collector)
                driver.get_screenshot_as_file(
                        './2-output-urls-data/%d-%s.png' % (i, downloader_folder[0]))
                f = open('./2-output-urls-data/%d-%s.json' % (i, downloader_folder[0]), 'w')
                f.write(json.dumps(el_data))
                f.close()

                count = 5
            except Exception as err:
                print(' - error - trying again')
                print(err)
                count = count + 1

        driver.quit()

sys.exit(0)
