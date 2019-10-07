salary = read.csv("Salary_Data.csv ")
set.seed(123)
library(caTools)
split = sample.split(salary$YearsExperience,SplitRatio = 2/3)
tester= subset(salary,split == FALSE)
trainer = subset(salary,split == TRUE)
regressor = lm(formula = Salary ~ YearsExperience,trainer)
Y_pred = predict(regressor, tester)
library(ggplot2)
ggplot() +
  geom_point(aes(trainer$YearsExperience,trainer$Salary),
             color = 'red') +
  geom_line(aes(trainer$YearsExperience,predict(regressor,trainer)),
            color = 'green') + 
  geom_point(aes(tester$YearsExperience,tester$Salary),
             color = 'black') +
  ggtitle('Salary vs Experience (Training set)') +
  xlab('Years of experience') +
  ylab('Salary')