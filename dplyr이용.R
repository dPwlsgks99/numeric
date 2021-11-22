text <- c("Crash dieting is not the best way to lose weight.",
          "A vegetariean diet excludes all animal flesh(meat, poultry, seafood).",
          "Economists surveyed by Refinitive expect the economy added 160,000 jobs.")
library(tm)
data(crude)
crude

crude[[1]]
crude[[1]]$content
crude[[1]]$meta

VCorpus()
getSources()

corpus.docs <- VCorpus(VectorSource(text))
class(corpus.docs)

corpus.docs
inspect(corpus.docs[1])
inspect(corpus.docs[[1]])

as.character(corpus.docs[[1]])
lapply(corpus.docs, as.character)

str(corpus.docs[[1]])

corpus.docs[[1]]$content
lapply(corpus.docs, content)
paste(as.vector(unlist(lapply(corpus.docs, content))), collapse=" ")

corpus.docs[[1]]$meta
meta(corpus.docs[[1]])

meta(corpus.docs[[1]], tag="id")
meta(corpus.docs[[1]], tag="datetimestamp")

meta(corpus.docs[[1]], tag="author")
meta(corpus.docs[[1]], tag="author", type="local")<-"BBC"
meta(corpus.docs[[1]])

source <- c("BBC", "CNN", "FOX")
meta(corpus.docs, tag="author", type="local") <- source
lapply(corpus.docs, meta, tag="author")

category <- c("health", "lifestyle", "business")
meta(corpus.docs, tag="category", type="local") <- category
lapply(corpus.docs, meta, tag="category")

meta(corpus.docs, tag="origin", type="local") <- NULL
lapply(corpus.docs, meta)

corpus.docs.filtered <-
  tm_filter(corpus.docs,
            FUN=function(x)
              any(grep("weight|diet", content(x))))
lapply(corpus.docs.filtered, content)

idx <- meta(corpus.docs, "author")=="FOX" | meta(corpus.docs, "category")=="health"
lapply(corpus.docs[idx], content)

writeCorpus(corpus.docs)
list.files(pattern="\\.txt")

getTransformations()

corpus.docs <- tm_map(corpus.docs, content_transformer(tolower))
lapply(corpus.docs, content)

stopwords("english")

corpus.docs <- tm_map(corpus.docs, removeWords, stopwords("english"))
lapply(corpus.docs, content)

myRemove <- content_transformer(function(x, pattern)
  {return(gsub(pattern,"", x))})

corpus.docs <- tm_map(corpus.docs, myRemove, "(f|ht)tp\\s+\\s")
lapply(corpus.docs, content)

corpus.docs <- tm_map(corpus.docs, removePunctuation)
lapply(corpus.docs, content)corpus.docs <- tm_map(corpus.docs, removePunctuation)
lapply(corpus.docs, content)

corpus.docs <- tm_map(corpus.docs, removeNumbers)
lapply(corpus.docs, content)

corpus.docs <- tm_map(corpus.docs, stripWhitespace)
lapply(corpus.docs, content)

corpus.docs <- tm_map(corpus.docs, content_transformer(trimws))
lapply(corpus.docs, content)

install.packages("SnowballC")
library(SnowballC)
corpus.docs <- tm_map(corpus.docs, stemDocument) # ��� ����
lapply(corpus.docs, content)


corpus.docs <- tm_map(corpus.docs, content_transformer(gsub), pattern="economist", replacement="economi")
lapply(corpus.docs, content)



text <- c("Crash dieting is not the best way to lose weight. http://bbc.in/1G0j4Agg",
          "A vegetariean diet excludes all animal flesh(meat, poultry, seafood).",
          "Economists surveyed by Refinitive expect the economy added 160,000 jobs.")
source <- c("BBC", "CNN","FOX")

library(dplyr)
text.df <- tibble(source=source, text=text)
text.df
class(text.df)

install.packages("tidytext")
library(tidytext)
unnest_tokens(tbl=text.df, output=word, input=text)

head(iris)
iris %>% head()

tidy.docs <- text.df %>%
  unnest_tokens(output=word, input=text)
tidy.docs
print(tidy.docs, n=Inf)

tidy.docs %>%
  count(source) %>%
  arrange(desc(n))

word.removed <- tibble(word=c("http", "bbc.in", "1G0j4agg"))
tidy.docs <- tidy.docs %>%
  anti_join(word.removed, by="word")
tidy.docs$word


grep("\\d+", tidy.docs$word)
tidy.docs <- tidy.docs[-grep("\\d+", tidy.docs$word),]
tidy.docs$word

text.df$text <- gsub("(f|ht)tp\\S+\\s*", "", text.df$text)
text.df$text <- gsub("\\d+", "", text.df$text)
text.df$text

tidy.docs <- text.df %>%
  unnest_tokens(output=word, input=text)
print(tidy.docs, n=Inf)

stop_words

tidy.docs <- tidy.docs %>%
  anti_join(stop_words, by="word")
tidy.docs$word

tidy.docs$word <- gsub("\\s+","", tidy.docs$word)

tidy.docs <- tidy.docs %>%
  mutate(word=wordStem(word))
tidy.docs$word

tidy.docs$word <- gsub("economist","economi", tidy.docs$word)
tidy.docs$word

corpus.docs <- VCorpus(VectorSource(text))
corpus.docs
meta(corpus.docs, tag="author", type="local") <- source
tidy(corpus.docs)

tidy(corpus.docs) %>%
  unnest_tokens(word, text) %>% 
  select(source=author, word)



text <- c("Crash dieting is not the best way to lose weight. http://bbc.in/1G0j4Agg",
          "A vegetariean diet excludes all animal flesh(meat, poultry, seafood).",
          "Economists surveyed by Refinitive expect the economy added 160,000 jobs.")

corpus.docs <- VCorpus(VectorSource(text))
corpus.docs <- tm_map(corpus.docs, content_transformer(tolower))
corpus.docs <- tm_map(corpus.docs, removeWords, stopwords("english"))
myRemove <- content_transformer(function(x, pattern)
{return(gsub(pattern,"", x))})
corpus.docs <- tm_map(corpus.docs, myRemove, "(f|ht)tp\\s+\\s")
corpus.docs <- tm_map(corpus.docs, removePunctuation)
corpus.docs <- tm_map(corpus.docs, removeNumbers)
corpus.docs <- tm_map(corpus.docs, stripWhitespace)
corpus.docs <- tm_map(corpus.docs, content_transformer(trimws))
corpus.docs <- tm_map(corpus.docs, stemDocument) # ��� ����
corpus.docs <- tm_map(corpus.docs, content_transformer(gsub), pattern="economist", replacement="economi")

corpus.docs
corpus.dtm <- DocumentTermMatrix(corpus.docs, control=list(wordLengths=c(2,Inf)))
corpus.dtm

nTerms(corpus.dtm)
Terms(corpus.dtm)

nDocs(corpus.dtm)
Docs(corpus.dtm)

rownames(corpus.dtm)
rownames(corpus.dtm) <- c("BBC", "CNN", "FOX")
Docs(corpus.dtm)

inspect(corpus.dtm)
inspect(corpus.dtm[1:2, 10:15])

tidy(corpus.dtm)
text <- c("Crash dieting is not the best way to lose weight. http://bbc.in/1G0j4Agg",
          "A vegetariean diet excludes all animal flesh(meat, poultry, seafood).",
          "Economists surveyed by Refinitive expect the economy added 160,000 jobs.")
source <- c("BBC", "CNN","FOX")
text.df <- tibble(source=source, text=text)
text.df$text <- gsub("(f|ht)tp\\S+\\s*", "", text.df$text)
text.df$text <- gsub("\\d+", "", text.df$text)
tidy.docs <- text.df %>%
  unnest_tokens(output=word, input=text) %>% 
  anti_join(stop_words, by="word") %>%
  mutate(word=wordStem(word))
tidy.docs$word <- gsub("\\s+","", tidy.docs$word)
tidy.docs$word <- gsub("economist","economi", tidy.docs$word)

tidy.docs %>% print(n=Inf)

tidy.docs %>%
  count(source, word)

tidy.dtm <- tidy.docs %>%
  count(source, word) %>%
  cast_dtm(document=source, term=word, value=n)
tidy.dtm

Terms(tidy.dtm)
Docs(tidy.dtm)
inspect(tidy.dtm)


install.packages("quanteda")
library(quanteda)
data_corpus_inaugural
summary(data_corpus_inaugural)
class(data_corpus_inaugural)

library(tidytext)
library(tibble)
library(dplyr)
tidy(data_corpus_inaugural) %>%
  filter(Year > 1990)

us.president.address <- tidy(data_corpus_inaugural) %>%
  filter(Year > 1990) %>%
  group_by(President, FirstName) %>%
  summarise_all(list(~trimws(paste(., collapse=" ")))) %>%
  arrange(Year) %>%
  ungroup()
us.president.address

library(tm)
us.president.address <- us.president.address %>%
  select(text, everything()) %>%
  add_column(doc_id=1:nrow(.), .before=1)
us.president.address
address.corpus <- VCorpus(DataframeSource(us.president.address))
address.corpus

lapply(address.corpus[1], content)

address.corpus <- tm_map(address.corpus, content_transformer(tolower))
lapply(address.corpus[1], content)

sort(stopwords("english"))

mystopwords <- c(stopwords("english"), c("can","must","will"))
address.corpus <- tm_map(address.corpus, removeWords, mystopwords)
address.corpus <- tm_map(address.corpus, removePunctuation)
address.corpus <- tm_map(address.corpus, removeNumbers)
address.corpus <- tm_map(address.corpus, stripWhitespace)
address.corpus <- tm_map(address.corpus, content_transformer(trimws))
lapply(address.corpus[1], content)

address.corpus <- tm_map(address.corpus, content_transformer(gsub),
                         pattern="america|americas|american|americans",
                         replacement="america")
lapply(address.corpus[1], content)

address.dtm <- DocumentTermMatrix(address.corpus)
inspect(address.dtm)

termfreq <- colSums(as.matrix(address.dtm))
length(termfreq)

termfreq[head(order(termfreq, decreasing = TRUE))]
termfreq[tail(order(termfreq, decreasing = TRUE))]

findFreqTerms(address.dtm, lowfreq = 40)
findFreqTerms(address.dtm, lowfreq = 50, highfreq = 100)

library(ggplot2)
termfreq.df <- data.frame(word=names(termfreq), frequency=termfreq)
head(termfreq.df)

ggplot(subset(termfreq.df, frequency >=40),
       aes(x=word, y=frequency, fill=word)) +
  geom_col(color="dimgray") +
  labs(x=NULL, y="Term Frequency (count)")

ggplot(subset(termfreq.df, frequency >=40),
       aes(x=word, y=frequency, fill=word)) +
  geom_col(color="dimgray", width=0.6, show.legend=FALSE) +
  geom_text(aes(label=frequency), size=3.5, color="black") +
  labs(x=NULL, y="Term Frequency (count)") +
  coord_flip()

set.seed(123)
library(wordcloud)
library(RColorBrewer)
head(termfreq)
wordcloud(words=names(termfreq), freq=termfreq,
          scale=c(4,0.5), min.freq = 5,
          rot.per=0.1, random.order=FALSE,
          colors=brewer.pal(6, "Dark2"),
          random.color = FALSE)
inspect(address.dtm)
rownames(address.dtm) <- c("Clinton", "Bush", "Obama", "Trump")
Docs(address.dtm)


address.tf <- tidy(address.dtm)
address.tf
address.tf <- address.tf %>%
  mutate(document=factor(document, levels=c("Clinton", "Bush", "Obama", "Trump"))) %>%
  arrange(desc(count)) %>%
  group_by(document) %>%
  top_n(n=10, wt=count) %>%
  ungroup()
address.tf

ggplot(address.tf,
       aes(x=term, y=count, fill=document)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~document, ncol=2, scales="free") +
  labs(x=NULL, y="Term Frequency (count)") +
  coord_flip()

ggplot(address.tf,
       aes(reorder_within(x=term, by=count, within=document), y=count, fill=document)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~document, ncol=2, scales="free") +
  scale_x_reordered() +
  labs(x=NULL, y="Term Frequency (count)") +
  coord_flip()

address.dtm2 <- DocumentTermMatrix(address.corpus, control=list(weighting=weightTfIdf))
rownames(address.dtm2) <- c("Clinton", "Bush", "Obama", "Trump")
Docs(address.dtm2)
inspect(address.dtm2)

tidy(address.dtm2)

address.tfidf <- tidy(address.dtm2) %>%
  mutate(tf_idf=count, count=NULL)
address.tfidf


address.tfidf <- address.tfidf %>%
  mutate(document=factor(document, levels=c("Clinton", "Bush", "Obama", "Trump"))) %>%
  arrange(desc(tf_idf)) %>%
  group_by(document) %>%
  top_n(n=10, wt=tf_idf) %>%
  ungroup()
address.tfidf

ggplot(address.tfidf,
       aes(reorder_within(x=term, by=tf_idf, within=document), y=tf_idf, fill=document)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~document, ncol=2, scales="free") +
  scale_x_reordered() +
  labs(x=NULL, y="Term Frequency-Inverse Document Frequency") +
  coord_flip()
us.president.address


address.words <- us.president.address %>%
  unnest_tokens(word, text)
address.words

address.words <- address.words %>%
  anti_join(stop_words, by="word") %>%
  filter(!grepl(pattern = "\\d+", word)) %>%
  mutate(word=gsub(pattern="'", replacement="", word)) %>%
  mutate(word=gsub(pattern="america|americas|american|americans", replacement="america", word)) %>%
  count(President, word, sort= TRUE, name="count") %>%
  ungroup()
address.words

address.words %>%
  group_by(word) %>%
  summarise(count=sum(count)) %>%
  arrange(desc(count)) %>%
  top_n(n=10, wt=count) %>%
  ggplot(aes(reorder(word, -count), count)) +
  geom_col(color="dimgray", fill="salmon", width=0.6, show.legend = FALSE ) +
  theme(axis.text.x=element_text(angle=45, hjust=1)) +
  geom_text(aes(label=count), size=3.5, color="black", vjust=-0.3) +
  labs(x=NULL, y="Term Frequency (count)") 

address.words <- address.words %>%
  bind_tf_idf(term=word, document=President, n=count)
address.words

address.words %>%
  arrange(desc(tf_idf))
address.words %>%
  arrange(tf_idf)

address.words %>%
  arrange(desc(tf_idf)) %>%
  mutate(President=factor(President, levels=c("Clinton", "Bush", "Obama", "Trump"))) %>%
  group_by(President) %>%
  top_n(7, wt=tf_idf) %>%
  ungroup() %>%
  ggplot(aes(reorder_within(word, tf_idf, President), tf_idf, fill=President)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~President, ncol=2, scales="free") +
  scale_x_reordered() +
  labs(x=NULL, y="Term Frequency-Inverse Document Frequency") +
  coord_flip()
