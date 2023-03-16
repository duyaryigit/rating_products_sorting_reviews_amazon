
###################################################
# PROJECT: Rating Product & Sorting Reviews in Amazon
###################################################

###################################################
# Business Problem
###################################################

# One of the most important problems in e-commerce is the correct calculation of the points given to the products after sales.
# The solution to this problem means providing greater customer satisfaction for the e-commerce site, prominence of the product for the sellers and a seamless shopping experience for the buyers.
# Another problem is the correct ordering of the comments given to the products.
# Since misleading comments will directly affect the sale of the product, it will cause both financial loss and loss of customers.
# In the solution of these 2 basic problems, while the e-commerce site and the sellers will increase their sales,
# the customers will complete the purchasing journey without any problems.

###################################################
# Dataset Story
###################################################

# This dataset, which includes Amazon product data, includes product categories and various metadata.
# The product with the most reviews in the electronics category has user ratings and reviews.

# reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
# asin - ID of the product, e.g. 0000013714
# reviewerName - name of the reviewer
# helpful - helpfulness rating of the review, e.g. 2/3
# reviewText - text of the review
# overall - rating of the product
# summary - summary of the review
# unixReviewTime - time of the review (unix time)
# reviewTime - time of the review (raw)
# day_diff - Number of days since assessment
# helpful_yes - The number of times the evaluation was found useful
# total_vote - Number of votes given to the evaluation


import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 10)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

###################################################
# Task 1: Calculate Average Rating Based on Current Comments and Compare with Existing Average Rating.
###################################################

# In the shared data set, users gave points and comments to a product.
# Our aim in this task is to evaluate the scores given by weighting them by date.
# It is necessary to compare the first average score with the weighted score according to the date to be obtained.


###################################################
# 1: Read the Data Set and Calculate the Average Score of the Product.
###################################################

df = pd.read_csv("datasets/amazon_review.csv")
df.head()
df["overall"].mean()
df["overall"].describe()
df["overall"].value_counts()
df["overall"].hist()
plt.show()


###################################################
# 2: Calculate the Weighted Average of Score by Date.
###################################################

# day_diff: How long has it been since the comment
df['reviewTime'] = pd.to_datetime(df['reviewTime'], dayfirst=True)
current_date = pd.to_datetime('2014-12-08 00:00:00')
df["day_diff1"] = (current_date - df['reviewTime']).dt.days
df["day_diff1"].describe()


# Determination of time-based average weights

q1 = df["day_diff"].quantile(0.25) # 281
q2 = df["day_diff"].quantile(0.50) # 431
q3 = df["day_diff"].quantile(0.75) # 601

# Calculate the weighted score according to the a, b, c values.

weighted_average=df.loc[df["day_diff"]<= q1, "overall"].mean()*50/100+\
                 df.loc[(df["day_diff"]> q1) & (df["day_diff"]<= q2), "overall"].mean()*25/100+\
                 df.loc[(df["day_diff"]> q2) & (df["day_diff"]<= q3), "overall"].mean()*15/100+\
                 df.loc[df["day_diff"]> q3, "overall"].mean()*10/100

# 3: Compare and interpret the average of each time period in weighted scoring.

df.loc[df["day_diff"]<= q1, "overall"].mean() # 4.70
df.loc[(df["day_diff"]> q1) & (df["day_diff"]<= q2) , "overall"].mean() # 4.63
df.loc[(df["day_diff"]> q2) & (df["day_diff"]<= q3) , "overall"].mean() # 4.57
df.loc[df["day_diff"]> q3, "overall"].mean() # 4.44

# Functionalize the process

def time_based_weighted_average(dataframe, w1=50, w2=25, w3=15, w4=10):
    return dataframe.loc[dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.25), "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.25)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.50)), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.50)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.75)), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.75)), "overall"].mean() * w4 / 100


time_based_weighted_average(df)

###################################################
# Task 2: Specify 20 Reviews for the Product to be Displayed on the Product Detail Page.
###################################################


###################################################
# 1. Generate the helpful_no variable
###################################################

# total_vote is the total number of up-downs given to a comment.
# up means helpful.
# There is no helpful_no variable in the dataset, it must be generated over existing variables.

df["helpful"].head()

# 1st Solution

df["helpful"]=df["helpful"].str.strip('[ ]')
df["helpful_yes"]=df["helpful"].apply(lambda x:x.split(", ")[0]).astype(int)
df["total_vote"]=df["helpful"].apply(lambda x:x.split(", ")[1]).astype(int)
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.head(50)

# 2nd Solution

df["helpful_yes"]=df[["helpful"]].applymap(lambda x:x.split(", ")[0].strip('[')).astype(int)
df["total_vote"]=df[["helpful"]].applymap(lambda x:x.split(", ")[1].strip(']')).astype(int)

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

df = df[["reviewerName", "overall", "summary", "helpful_yes", "helpful_no", "total_vote", "reviewTime"]]

###################################################
# 2. Calculate score_pos_neg_diff, score_average_rating and wilson_lower_bound scores and add to data
###################################################

df.head()

# score_pos_neg_diff

def score_up_down_diff(up, down):
    return up - down

df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

# 2nd Solution

df["score_pos_neg_diff"]=df["helpful_yes"]- df["helpful_no"]


# score_average_rating

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)


def wilson_lower_bound(up, down, confidence=0.95):
    """
    Calculate Wilson Lower Bound Score

    - The lower limit of the confidence interval to be calculated for the Bernoulli parameter p is accepted as the WLB score.
    - The score to be calculated is used for product ranking.
    - Note:
    If the scores are between 1-5, 1-3 are marked as negative, 4-5 as positive and can be made to conform to Bernoulli.
    his brings with it some problems. For this reason, it is necessary to make a bayesian average rating.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

df["wilson_lower_bound"].describe([0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.77,0.95,0.97,0.98,0.99,1])
df["wilson_lower_bound"].value_counts()


##################################################
# 3. Identify 20 Interpretations and Interpret Results.
###################################################

df.sort_values("wilson_lower_bound", ascending=False).head(20)
