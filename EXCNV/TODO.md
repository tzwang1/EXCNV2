TODO
Add all pickle files together after job finishes running
Try conifer on ccbr cluster - fails
Try cnvkit on gcloud
Make sure input output generation finishes - done

Why doesn't this method work?
Generic reasons
1. Data is too noisy (too much bias)
2. Not enough samples
3. The data is not suitable for cnv detection - if we have a target 100 200 gain, it could be that a region outside of this segment is when the CNV changed from neutral to gain, however we have no data on this.
4. 

Specific reasons
1. Log R ratio is calculated by comparing a cancer sample with a control sample
    - the control for each sample can be different
    - To overcome this we could give the model a reference sample as well

How can my current method be improved
1. Increase the data size
2. Better data (not always possible)
3. Use Google's Deep Variant to try and improve the accuracy of the data
4. Try to engineer more features other than GC content and read depth
5. Could try using CBS to separate data into segments first and then use thos specific segments as input data, instead of the whole exome