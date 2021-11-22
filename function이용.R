library(tidytext)
install.packages("textdata")
library(textdata)

get_sentiments(lexicon = "bing")
unique(get_sentiments("bing")$sentiment)

get_sentiments(lexicon = "afinn")
unique(get_sentiments(lexicon = "afinn")$value)
summary(get_sentiments("afinn")$value)

get_sentiments(lexicon="nrc")
unique(get_sentiments(lexicon="nrc")$sentiment)

get_sentiments(lexicon="loughran")
unique(get_sentiments(lexicon="loughran")$sentiment)

library(dplyr)
library(tibble)
library(purrr)
library(readr)
library(lubridate)

url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00438/Health-News-Tweets.zip"
local.copy <- tempfile()
download.file(url, local.copy, mode="wb")
Sys.setlocale("LC_TIME","English")

health.twitter <- map(unzip(zipfile=local.copy,
                            files=c("Health-Tweets/bbchealth.txt",
                                    "Health-Twwets/cnnhealth.txt",
                                    "Health-Tweets/foxnewshealth.txt",
                                    "Health-Tweets/NBChealth.txt")),
                      read_delim, delim="|", quote="",
                      col_types=list(col_character(), col_character(), col_character()),
                                     col_names=c("id","datetime","tweet")) %>%
                        map2(c("bbc","cnn","foxnews","nbc"),
                             ~ cbind(.x, source=.y)) %>%
                        reduce(bind_rows) %>%
                        as_tibble() %>%
                        mutate(datetime=ymd_hms(strptime(datetime, "%a %b %d %H:%M:%S + 0000 %Y")))

health.twitter
health.twitter %>% count(source)

library(stringr)
health.words <- health.twitter %>% select(-id) %>%
  mutate(tweet=str_replace_all(tweet, pattern="(f|ht)tp\\S+S*", replacement="")) %>%
  mutate(tweet=str_replace_all(tweet, pattern="\\d+", replacement="")) %>%
  mutate(tweet=str_replace_all(tweet, pattern="\\bRT", replacement="")) %>%
  mutate(tweet=str_replace_all(tweet, pattern="@\\S+", replacement = "")) %>%
  mutate(tweet=str_replace_all(tweet, pattern="&amp", replacement="")) %>%
  unnest_tokens(word, tweet)

health.words

health.words %>% inner_join(get_sentiments("bing"), by="word")
health.words %>% inner_join(get_sentiments("bing"), by="word") %>%
  count(word, sentiment, sort=TRUE)
health.sentiment <- health.words %>% inner_join(get_sentiments("bing"), by="word") %>%
  count(word, sentiment, sort=TRUE) %>% 
  group_by(sentiment) %>%
  top_n(10,n) %>% 
  ungroup() %>%
  mutate(nsign=ifelse(sentiment=="negative", -n, n))
health.sentiment

library(ggplot2)
library(scales)


ggplot(health.sentiment,
       aes(x=reorder(word, nsign), y=nsign,
           fill=factor(sentiment, levels=c("positive", "negative"))))+
  geom_col(color="lightslategray", width=0.8) +
  geom_text(aes(label=n), size=3, color="black",
            hjust=ifelse(health.sentiment$nsign<0, 1.1, -0.1)) +
  scale_fill_manual(values=c("cornflowerblue", "tomato")) +
  scale_y_continuous(breaks=pretty(health.sentiment$nsign),
                     labels=abs(pretty(health.sentiment$nsign))) +
  labs(X=NULL, y="Count")+
  theme(legend.position = "bottom", legend.title = element_blank()) +
  coord_flip()

health.sentiment <- health.words %>% inner_join(get_sentiments("bing"), by="word") %>%
  filter(!(word=="patient"| word=="cancer"| word=="virus")) %>%
  count(word, sentiment, sort=TRUE) %>% 
  group_by(sentiment) %>%
  top_n(10,n) %>% 
  ungroup() %>%
  mutate(nsign=ifelse(sentiment=="negative", -n, n))
health.sentiment

ggplot(health.sentiment,
       aes(x=reorder(word, n), y=n,
           fill=factor(sentiment, levels=c("positive", "negative"))))+
  geom_col(color="lightslategray", width=0.8, show.legend = ) +
  geom_text(aes(label=n), size=3, color="black",
            hjust=ifelse(health.sentiment$nsign<0, 1.1, -0.1)) +
  scale_fill_manual(values=c("lightsteelblue", "lightsalmon")) +
  facet_wrap(~ factor(sentiment, levels=c("positive", "negative")),
             ncol=2,scales="free") +
  labs(X=NULL, y="Count")+
  theme(legend.position = "bottom", legend.title = element_blank()) +
  coord_flip()

library(wordcloud)
library(reshape2)
set.seed(123)
health.words %>% inner_join(get_sentiments("bing"), by="word") %>%
  filter(!(word=="patient"| word=="cancer"| word=="virus")) %>%
  count(word, sentiment, sort=TRUE) %>% 
  ungroup() %>%
  acast(word ~ sentiment, value.var="n", fill=0) %>% 
  comparison.cloud(colors=c("tomato", "cornflowerblue"),
                   title.size=2,
                   title.colors=c("red","blue"),
                   title.bg.colors = "wheat",
                   scale=c(4,0.3), max.words=200)

health.sentiment <- health.words %>% inner_join(get_sentiments("bing"), by="word") %>%
  filter(!(word=="patient"| word=="cancer"| word=="virus")) %>%
  count(word, sentiment, sort=TRUE) %>% 
  group_by(sentiment) %>%
  top_n(10,n) %>% 
  ungroup()
  #mutate(nsign=ifelse(sentiment=="negative", -n, n))
health.sentiment

windows(width=7, height=9)
ggplot(health.sentiment,
       aes(reorder_within(x=word, by=n, within=source),
           y=n, fill=source)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ factor(source,
                      labels=c("BBC", "CNN", "Fox News","NBC")) +
               sentiment, ncol=2, scales="free") +
  scale_x_reordered() +
  labs(x=NULL, y="Count") +
  coord_flip()

health.sentiment <- health.words %>% inner_join(get_sentiments("bing"), by="word") %>%
  filter(!(word=="patient"| word=="cancer"| word=="virus")) %>%
  mutate(time=floor_date(x=datetime, unit="month")) %>%
  count(sentiment, time) %>% 
  group_by(sentiment) %>%
  slice(2:(n()-1)) %>%
  ungroup()
health.sentiment

Sys.setlocale("LC_TIME", "English")
windows(width=7.0, height=5.5)
ggplot(health.sentiment, aes(x=time, y=n, fill=sentiment, color=sentiment)) +
  geom_area(position="identity", alpha=0.3) +
  geom_line(size=1.5) +
  scale_fill_manual(labels=c("Negative", "Positive"),
                    values=c("orangered", "deepskyblue2")) +
  scale_x_datetime(date_labels="%b %Y", date_breaks = "6 months") +
  labs(x=NULL, y="Count") +
  theme(legend.position = "bottom", legend.title=element_blank())


### text mining-classific analytics ###
url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
local.copy <- tempfile()
download.file(url, local.copy, mode="wb")
library(readr)
sms <- read_delim(unzip(zipfile=local.copy, files="SMSSpamCollection"),
                  delim="\t", quote="",
                  col_types=cols("f","c"),
                  col_names=c("type","text"))
unlink(local.copy)
sms

table(sms$type)
prop.table(table(sms$type))

library(dplyr)
library(tibble)

sms <- sms %>% 
  select(text, type) %>%
  add_column(doc_id=1:nrow(.), .before=1) %>%
  mutate(text=iconv(text, to="ascii", sub=""))

sms

library(tm)
docs <- VCorpus(DataframeSource(sms))

#lapply(docs, content)
lapply(docs, content)[c(13,16,20)]
meta(docs)
meta(docs)$type[c(13,16,20)]

docs <- tm_map(docs, content_transformer(tolower))
lapply(docs,content)[c(13,16,20)]


myremove <- content_transformer(function(x,pattern)
  {return(gsub(pattern, "",x))})
docs <- tm_map(docs, myremove, "(f|ht)tp\\S+\\s*")
docs <- tm_map(docs, myremove, "www\\.+\\S+")
lapply(docs, content)[c(13,16,20)]

mystopwords <- c(stopwords("english"),
                 c("can","cant","don","dont","get","got","just","one","will"))
docs <- tm_map(docs, removeWords, mystopwords)

tospace <- content_transformer(function(x,pattern)
  {return(gsub(pattern, " ",x))})
docs <- tm_map(docs, tospace, ":")
docs <- tm_map(docs, tospace, ";")
docs <- tm_map(docs, tospace, "/")
lapply(docs, content)[c(13,16,20)]


docs <- tm_map(docs, removePunctuation)
docs <- tm_map(docs, removeNumbers)
docs <- tm_map(docs, stripWhitespace)
docs <- tm_map(docs, content_transformer(trimws))
docs <- tm_map(docs, stemDocument)
lapply(docs, content)[c(13,16,20)]

dtm <- DocumentTermMatrix(docs)
dtm
inspect(dtm)


dtm <- DocumentTermMatrix(docs, control=list(wordLengths=c(4,10), bounds=list(global=c(5,5300))))
dtm

termfreq <- colSums(as.matrix(dtm))
head(termfreq)

termfreq[head(order(termfreq, decreasing=TRUE))]
termfreq[tail(order(termfreq, decreasing=TRUE))]

findFreqTerms(dtm, lowfreq=200)

findAssocs(dtm, c("call","free"),c(0.20, 0.25))

library(wordcloud)   
install.packages("RColorBrewer")
library(RColorBrewer)

set.seed(123)
windows(width=7, height=7)
wordcloud(words=names(termfreq), freq=termfreq,
          scale=c(4,0.5), min.freq = 30, max.words = 200,
          rot.per=0, random.order = FALSE, random.color=FALSE,
          colors=brewer.pal(6,"Set2"))


hamspam <- as.matrix(dtm)
rownames(hamspam) <- sms$type
hamspam[1:5, 1:5]

hamspam <- rowsum(hamspam, group=rownames(hamspam))
hamspam[,1:5]

set.seed(123)
windows(width=7, height=7)
comparison.cloud(t(hamspam),
                 colors=c("cornflowerblue", "tomato"),
                 title.size = 2, title.colors = c("blue", "red"),
                 title.bg.colors = "wheat",
                 rot.per = 0, scale=c(5,0.4), max.words=200)

inspect(dtm)
sms$type

set.seed(123)
train <- sample(nrow(sms), 0.7*nrow(sms))
y.train <- sms[train,]$type
y.test <- sms[-train,]$type
table(y.train)
table(y.test)

prop.table(table(y.train))
prop.table(table(y.test))

tofactor <- function(x){
  x<-ifelse(x>0,1,0)
  x<-factor(x,level=c(0,1), labels=c("no","yes"))
  return(x)
}

sms.dtm <- apply(dtm, MARGIN=2, tofactor)
str(sms.dtm)
sms.dtm[1:5, 1:5]


x.train <- sms.dtm[train,]
x.test <- sms.dtm[-train,]

library(e1071)
sms.nb <- naiveBayes(x=x.train, y=y.train)
sms.nb.pred <- predict(sms.nb, newdata=x.test)
head(sms.nb.pred)

table(y.test, sms.nb.pred, dnn=c("Actual", "Predicted"))
mean(sms.nb.pred==y.test)
