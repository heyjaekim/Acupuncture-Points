d = seq(1,100,1)

s1 = sample(d, 20)
s2 = sample(d, 20)
sum(s1 == s2)
s3 = sample(d,20)
sum(s3 == c(s1, s2))

s4 = sample(d,20)
sum(s4 == c(s1, s2, s3))

s5 = sample(d,20)
sum(s5 == c(s1, s2, s3, s4))

intersect(s5, c(s1, s2, s3, s4))

d
simulation <- function (rep =100, sample_size = 20, thresh = 95){
  result <- c()
  overlap_len <- c()
  for (i in 1:rep){
    s = sample(d, 20)
    overlap = intersect(s, result)
    len_overlap = length(overlap)
    overlap_len <- c(overlap_len, len_overlap)
#   print(len_overlap /sample_size)
    result <- c(result, s)
    #print(length(unique(result)))
    if ( length(unique(result)) > thresh ){
    #  cnt <- i
      break
    }
   # print(as.integer(summary(table(result))[2]))
   # print(summary(table(result)))
  }
  return (list(i, overlap_len, length(unique(result)), table(result))) 
}
simulation()[[1]]
loop_inds <- c()
avg_loops <- c()
for (i in 1:30){
  loop_inds <- c(loop_inds, simulation(thresh = 95)[[1]] )
  avg_loops <- c( avg_loops, mean(loop_inds))
}
loop_inds
avg_loops
