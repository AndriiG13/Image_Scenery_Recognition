#############################################################################################################
#
# Kaggle 4: Scene Recognition
# Andrii and Vincent: Run Forest Run
# 
# This script uses extreme boosting to optimize the scene recognition
#
#############################################################################################################
rm(list=ls())

library(tidyverse) # metapackage with lots of helpful functions
library(jpeg) # package that can be used to read jpeg image files in a simple format
library(e1071)
library(psd)
library(entropy)
library(modes)
library(xgboost)

skies = dir("images_split/cloudy_sky/", full.names = TRUE)
rivers = dir("images_split/rivers/", full.names = TRUE)
sunsets = dir("images_split/sunsets/", full.names = TRUE)
trees = dir("images_split/trees_and_forest/", full.names = TRUE)
test_set = dir("images_split/test_set/", full.names = TRUE)

readJPEG_as_df <- function(path, featureExtractor = I) {
  img = readJPEG(path)
  
  d = dim(img) 
  # add names to the array dimensions
  dimnames(img) = list(x = 1:d[1], y = 1:d[2], color = c('r','g','b')) 
  # turn array into a data frame 
  df  <- 
    as.table(img) %>% 
    as.data.frame(stringsAsFactors = FALSE) %>% 
    # make the final format handier and add file name to the data frame
    mutate(file = basename(path), x = as.numeric(x)-1, y = as.numeric(y)-1) %>%
    mutate(pixel_id = x + 28 * y) %>% 
    rename(pixel_value = Freq) %>%
    select(file, pixel_id, x, y, color, pixel_value)
  # extract features 
  df %>%
    featureExtractor
}


readEdges_as_df <- function(path, featureExtractor = I) {
  img = readJPEG(path)
  
  img <- edge_detection(img, method = "LoG", conv_mode = "same",
                        gaussian_dims = 3, sigma = 1, range_gauss = 2,
                        laplacian_type = 1)
  # image dimensions
  d = dim(img) 
  # add names to the array dimensions
  dimnames(img) = list(x = 1:d[1], y = 1:d[2], color = c('s','h','j')) 
  # turn array into a data frame 
  df  <- 
    as.table(img) %>% 
    as.data.frame(stringsAsFactors = FALSE) %>% 
    # make the final format handier and add file name to the data frame
    mutate(file = basename(path), x = as.numeric(x)-1, y = as.numeric(y)-1) %>%
    mutate(pixel_id = x + 28 * y) %>% 
    rename(pixel_value = Freq) %>%
    select(file, pixel_id, x, y, color, pixel_value)
  # extract features 
  df %>%
    featureExtractor
}






peekImage <- . %>% spread(color, pixel_value) %>%
  mutate(x=rev(x), color = rgb(r,g,b)) %>%
  {ggplot(., aes(y, x, fill = color)) + geom_tile(show.legend = FALSE) + theme_light() + 
    scale_fill_manual(values=levels(as.factor(.$color))) + facet_wrap(~ file)}

xx <- readJPEG_as_df(rivers[2]) 
readJPEG_as_df(sunsets[1]) %>% peekImage

img = readJPEG(rivers[2])

##make a couple of functions for features
entropy  <- function(x, nbreaks = nclass.Sturges(x)) {
  r = range(x)
  x_binned = findInterval(x, seq(r[1], r[2], len= nbreaks))
  h = tabulate(x_binned, nbins = nbreaks) # fast histogram
  p = h/sum(h)
  -sum(p[p>0] * log(p[p>0]))
}

peak2peak <- function (x) { return (max(x) - min(x)) }

rms <- function (x) { return (sqrt(mean(x^2))) }

##create features
myFeatures <- function(file) {
  
  nr = nc = 3  # maybe try to change 
  file %>% # starting with '.' defines the pipe to be a function 
    group_by(file, X=cut(x, nr, labels = FALSE)-1, Y=cut(y, nc, labels=FALSE)-1, color) %>%
    summarise(
      m = mean(pixel_value),
      s = sd(pixel_value),
      min = min(pixel_value),
      max = max(pixel_value),
      q25 = quantile(pixel_value, .25),
      q75 = quantile(pixel_value, .75),
      p2p = peak2peak(pixel_value),
      rms = rms(pixel_value), 
      kurt = e1071::kurtosis(pixel_value), 
      skew1 = e1071::skewness(pixel_value),
      #cf = max(abs(pixel_value))/rms(pixel_value),
      #v_rms = rms(diffinv(as.vector(pixel_value))), 
      AR3 = cor(pixel_value, lag(pixel_value, n = 3), use = "pairwise"),
      AR2 = cor(pixel_value, lag(pixel_value, n = 2), use = "pairwise"),
      AR4 = cor(pixel_value, lag(pixel_value, n = 4), use = "pairwise"),
      entropy = entropy(discretize(pixel_value, numBins = 10 )), 
      bim = bimodality_coefficient(pixel_value), 
      fft = mean(abs(fft(pixel_value))), 
      psd = mean(spectrum(pixel_value, plot = F)$spec)
      
    ) 
}


##create features
myFeatures_edges <- function(file) {
  
  nr = nc = 3  # maybe try to change 
  file %>% # starting with '.' defines the pipe to be a function 
    group_by(file, X=cut(x, nr, labels = FALSE)-1, Y=cut(y, nc, labels=FALSE)-1, color) %>%
    summarise(
      m = mean(pixel_value),
      s = sd(pixel_value),
      AR3 = cor(pixel_value, lag(pixel_value, n = 3), use = "pairwise"),
      AR2 = cor(pixel_value, lag(pixel_value, n = 2), use = "pairwise"),
      AR4 = cor(pixel_value, lag(pixel_value, n = 4), use = "pairwise")
      
    ) 
}



##because we need to reshape from long to wide format multiple times lets define a function:
myImgDFReshape = . %>%
  gather(feature, value, -file, -X, -Y, -color) %>% 
  unite(feature, color, X, Y, feature) %>% 
  spread(feature, value)

readHog <- function(path) {
  
jp <-   readJPEG(path)
  h <- HOG(jp, cells = 3, orientations = 6)
  
  data.frame(stringsAsFactors = FALSE, file = basename(path), h = h, edg = paste0("edg", 1:length(h))) %>% 
    spread(key = edg,value = h)
}

Sunsets_hog = map_df(sunsets[1:length(sunsets)], readHog)

Trees_hog = map_df(trees[1:length(trees)], readHog)

Rivers_hog = map_df(rivers[1:length(rivers)], readHog)

Skies_hog = map_df(skies[1:length(skies)], readHog)

Train_hog <- bind_rows(Sunsets_hog, Trees_hog, Rivers_hog, Skies_hog) 


##make all the seperate files
Sunsets_edges = map_df(sunsets[1:length(sunsets)], readEdges_as_df, featureExtractor = myFeatures_edges) %>% 
  myImgDFReshape 
Trees_edges = map_df(trees[1:length(trees)], readEdges_as_df, featureExtractor = myFeatures_edges) %>% 
  myImgDFReshape 
Rivers_edges = map_df(rivers[1:length(rivers)], readEdges_as_df, featureExtractor = myFeatures_edges) %>% 
  myImgDFReshape 
Skies_edges = map_df(skies[1:length(skies)], readEdges_as_df, featureExtractor = myFeatures_edges) %>% 
  myImgDFReshape 

Train_edges <- bind_rows(Sunsets_edges, Trees_edges, Rivers_edges, Skies_edges) 

Train_edges <- dplyr::select(Train_edges, -category)

##make all the seperate files
Sunsets = map_df(sunsets[1:length(sunsets)], readJPEG_as_df, featureExtractor = myFeatures) %>% 
  myImgDFReshape %>%
  mutate(category = "sunsets")
Trees = map_df(trees[1:length(trees)], readJPEG_as_df, featureExtractor = myFeatures) %>% 
  myImgDFReshape %>%
  mutate(category = "trees_and_forest")
Rivers = map_df(rivers[1:length(rivers)], readJPEG_as_df, featureExtractor = myFeatures) %>% 
  myImgDFReshape %>%
  mutate(category = "rivers")
Skies = map_df(skies[1:length(skies)], readJPEG_as_df, featureExtractor = myFeatures) %>% 
  myImgDFReshape %>%
  mutate(category = "cloudy_sky")


Train_features <- bind_rows(Sunsets, Trees, Rivers, Skies) 


D <- inner_join(Train_hog, Train_edges, by = "file")
Train <- inner_join(D, Train_features, by = "file")

Train <- as_tibble(Train)


##make final training set
Train$category <- as.factor(Train$category)

Train <- Train %>% ungroup()

cors <- cor(dplyr::select(Train, -file, -category))

hc <- findCorrelation(cors, cutoff = .99, names = T)

Train_no_cor <- dplyr::select(Train, -one_of(hc))


##make function to filter features based on pvalues
filter_on_importance = function(data){
  
  r_fit_not_caret <- ranger(category ~ ., data=data[,-1], num.trees = 500, mtry = 15, 
                          min.node.size = 20, splitrule = "gini",
                          importance = "impurity_corrected")
  
  imp_pvalues <- importance_pvalues(r_fit_not_caret,method = "altmann", formula = category ~ ., 
                                  data = data[,-1])

  imps <- as.tibble(cbind(rownames(imp_pvalues), imp_pvalues))

  imps$pvalue <- as.numeric(imps$pvalue)

  sig_imps <- imps %>% 
    dplyr::filter(pvalue < 0.01)

  pvalues <- as.numeric(imp_pvalues[,2])

  features <- sig_imps$V1

  data <- data %>% 
    dplyr::select(one_of(c(features, "category")))

  return(data)
}
 
rf <- ranger(category ~ ., data = Train_subset, num.trees = 4000, min.node.size = 2, max.depth = 15,verbose = T, 
       splitrule = "gini", mtry = floor(sqrt(ncol(Train_subset))))
rf

##make two subsets to reduce amount of features twice
Train_subset <- filter_on_importance(Train_no_cor)

Train_subset_2 <- filter_on_importance(Train_subset)

##change the training set to be suitable for modelling
Train_labs <- as.numeric(Train_subset$category) - 1

new_train <- model.matrix(~ . + 0, data = dplyr::select(Train_subset, -category))

xgb_train <- xgb.DMatrix(data = new_train, label = Train_labs)

##set parameters of the model and do crossvalidation for optimal number of trees
params <- list(booster = "gbtree", objective = "multi:softprob", num_class = 4, eval_metric = "merror")

bst_slow = xgb.cv(data = xgb_train,
                  params = params,
                  max.depth=1, 
                  eta = 0.0005, 
                  nround = 100000, 
                  nfold = 5, 
                  early_stopping_rounds = 4000,
                  print_every_n = 30)

tc <- trainControl(method = "cv", number = 10, verboseIter = T)




bst_slow4 = xgb.cv(data = xgb_train,
                  params = params,
                  max.depth=4, 
                  eta = 0.001, 
                  nround = 100000, 
                  nfold = 5, 
                  early_stopping_rounds = 1000,
                  print_every_n = 30)



bst_slow_smaller = xgb.cv(data = xgb_train1,
                  params = params,
                  max.depth=1, 
                  eta = 0.001, 
                  nround = 100000, 
                  nfold = 5, 
                  early_stopping_rounds = 1000,
                  print_every_n = 30)

bst_slow4_smaller = xgb.cv(data = xgb_train,
                   params = params,
                   max.depth=4, 
                   eta = 0.001, 
                   nround = 100000, 
                   nfold = 5, 
                   early_stopping_rounds = 1000,
                   print_every_n = 30)


##make the model based on the best iteration from function above
boost_model = xgboost(data = xgb_train,
                      params = params,
                      max.depth = 4,
                      eta = 0.001,
                      nround = bst_slow4$best_iteration,
                      early_stopping_rounds = 1000,
                      print_every_n = 30)

boost_model_subs = xgboost(data = xgb_train,
                      params = params,
                      max.depth = 1,
                      eta = 0.001,
                      nround = bst_slow$best_iteration,
                      early_stopping_rounds = 1000,
                      print_every_n = 30,
                      subsample = .7)

##make the test dataset 
Test_features <- map_df(test_set[1:length(test_set)], readJPEG_as_df, featureExtractor = myFeatures) %>%
  myImgDFReshape

Test_edges <- map_df(test_set[1:length(test_set)], readEdges_as_df, featureExtractor = myFeatures_edges) %>% 
  myImgDFReshape 

Test_hog <-  map_df(test_set[1:length(test_set)], readHog)



TD <- inner_join(Test_edges, Test_hog)
Test <- inner_join(TD, Test_features)


Test <- ungroup(Test)

##keep the significant features 
Test_subset1 <- Test %>%
  dplyr::select(one_of(c(colnames(Train_subset))))

Test_subset2 <- Test %>%
  dplyr::select(one_of(c(features2)))

##predict
predictions <- predict(boost_model_subs, newdata = data.matrix(Test_subset1), reshape = T)

##adjust column names
colnames(Test_subset2) <- boost_model$feature_names

labels <- apply(predictions, 1, which.max)

##make a function for transforming probabilities to labels
number_to_picture <- function(x) {
  
  x[x==1] <- "cloudy_sky"
  x[x=="2"] <- "rivers"
  x[x=="3"] <- "sunsets"
  x[x=="4"] <- "trees_and_forest"
  
  return(x)
}

labels_pictures <- number_to_picture(labels)

##write submission to csv
Test %>% ungroup %>% transmute(file=file, category = labels_pictures) %>% write_csv("submissionboostedsub.csv")



p <- predict(rf, Test)



