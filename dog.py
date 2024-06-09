from icrawler.builtin import BingImageCrawler

# /* 犬の画像を100枚取得 */
crawler = BingImageCrawler(storage={"root_dir": "dog"})
crawler.crawl(keyword="犬", max_num=100)