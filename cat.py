from icrawler.builtin import BingImageCrawler

# /* 猫の画像を100枚取得 */
crawler = BingImageCrawler(storage={"root_dir": "cat"})
crawler.crawl(keyword="猫", max_num=100)