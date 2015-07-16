# GenDec
Detect gender by name and surname. Designed for cyrillic names (should work with transliteraded names as well), but may work and for other languages too.

Under the hood regularized linear regression with gradient descent.
Ideas for features by Alexander Panchenko: http://www.slideshare.net/alexanderpanchenko/panchenko

# Usage
For now it requires two files: test and train. Files should contain names and surnames: one pair per line, separated by space.
