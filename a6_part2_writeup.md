# Assignment 6 Part 2 - Writeup

---

## Question 1: Feature Importance

Based on your house price model, rank the four features from most important to least important. Explain how you determined this ranking.

**YOUR ANSWER:**
1. Age
2. Square Feet
3. Bedroom
4. Bathroom

**Explanation:**
The older the house the less valuable it is
The bigger the house the more valuable it is cause you have more space
More Bedrooms mean more room/sleeping places
Bathrooms are also very important to have a home, but Bedrooms are the staples of any home

---

## Question 2: Interpreting Coefficients

Choose TWO features from your model and explain what their coefficients mean in plain English. For example: "Each additional bedroom increases the price by $___"

**Feature 1:**
For every Bedroom, the price increases by about $5000

**Feature 2:**
For Every 200 Square Meters ina house, the price increases by about $1700

---

## Question 3: Model Performance

What was your model's R² score? What does this tell you about how well your model predicts house prices? Is there room for improvement?

**YOUR ANSWER:**
Feet vs. Price - .992
Bedrooms vs. Price - .895
Bathrooms vs. Price - .856
Age vs. Price - .100

---

## Question 4: Adding Features

If you could add TWO more features to improve your house price predictions, what would they be and why?

**Feature 1:**
Neighborhood

**Why it would help:**
Some neighborhoods have higher averages of price, then other neighborhoods 

**Feature 2:**
Color/Style of the house

**Why it would help:**
So people can find the hosue right for them. Plus we can see how different a mansion is from a ranch house

---

## Question 5: Model Trust

Would you trust this model to predict the price of a house with 6 bedrooms, 4 bathrooms, 3000 sq ft, and 5 years old? Why or why not? (Hint: Think about the range of your training data)

**YOUR ANSWER:**
No because it has no idea how to calculate 3000 sq ft house with over 10 rooms, and it would contradict itself.

