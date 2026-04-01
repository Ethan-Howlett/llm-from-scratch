# Import training run table (CSV)
csv_path <- '/media/volume/Ethan-s-Volume/llm/final_project/training-data.csv'
data <- read.csv(csv_path, stringsAsFactors = FALSE, check.names = TRUE)

# --- balanced (0/1) vs test_accuracy (numeric) ---
# H0: mean test_accuracy is the same for balanced=0 and balanced=1.
# Welch two-sample t-test (default: does not assume equal variances).
data$balanced_f <- factor(data$balanced, levels = c(0, 1), labels = c("unbalanced", "balanced"))
t.test(test_accuracy ~ balanced_f, data = data)

# --- epochs: compare test_accuracy across epoch counts ---
data$epochs_f <- factor(data$epochs)

# Descriptive means (full table)
aggregate(test_accuracy ~ epochs_f, data = data, FUN = function(x) c(mean = mean(x), n = length(x)))

# Mean test_accuracy by epoch (same rows as epoch_compare)
epoch_means <- aggregate(test_accuracy ~ epochs, data = epoch_compare, FUN = mean)
epoch_means <- epoch_means[order(epoch_means$epochs), ]

out_dir <- file.path(dirname(csv_path), "stats_output")
dir.create(out_dir, showWarnings = FALSE)
png(filename = file.path(out_dir, "epoch_vs_mean_test_accuracy.png"),
    width = 7, height = 5, units = "in", res = 120)
plot(epoch_means$epochs, epoch_means$test_accuracy,
     type = "b", pch = 19, lwd = 2,
     xlab = "Epochs", ylab = "Mean test accuracy",
     xaxt = "n")
axis(1, at = epoch_means$epochs)
dev.off()

# Non-parametric omnibus + pairwise (same rows as epoch_compare)
# kruskal.test(test_accuracy ~ epochs_f, data = epoch_compare)
# pairwise.wilcox.test(epoch_compare$test_accuracy, epoch_compare$epochs_f, p.adjust.method = "holm")
