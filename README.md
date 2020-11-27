# metagenomic_sequencing_dl
DNA sequencing has produced a wealth of information in the past 40 years, on organisms from all environments on earth and all branches of the tree of life. As sequencing costs continue to fall, the amount of information generated has only accelerated. Many projects are dedicated to understanding the microbes that live in the environment and inside our own bodies (the human microbiome).  Metagenomic sequencing is an unbiased method to study DNA from microbes in complex environments, such as the human gut microbiome. The ability to survey these organisms, in a theoretically unbiased way, is one of the major advantages of metagenomics. 

While some common human-associated microbes or pathogenic agents, such as E. coli,  have been studied extensively, most microbes in complex environments have never been isolated, cultured and sequenced in a laboratory before. This is the weakness of metagenomics - reference databases are incomplete. 

The easiest way to classify a novel DNA sequence is to align the sequence to a large reference database. Algorithms such as BLAST(ref) are well suited to do this. However, these “reference-based” approaches will fail if similar sequences as the query are not present in the database. As viruses are some of the most diverse and least-studied agents on earth, reference-based approaches are particularly weak in these areas. 
	
To circumvent the drawbacks of reference-based approaches, “reference-free” methods have been developed. These methods use sequence features that are not based on alignment to references, such as the frequency of tetranucleotides (4 letter DNA words) to classify and cluster novel sequences (ref). Reference-free methods attempt to capture underlying features present in a DNA sequence without aligning it to a database. Deep learning is incredibly well suited to this task. 
	
In this project, we have developed a deep learning model for the binary task of classifying a DNA sequence as viral or bacterial. 
