"""Build the PDF and LaTeX report from saved experiment results."""
import argparse,json,html,re
from pathlib import Path
import pandas as pd
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--author',default='Alex Dems');ap.add_argument('--repo-url',default='');args=ap.parse_args()
    if args.repo_url and not args.repo_url.startswith(('https://github.com/','https://gitlab.com/')):
        raise ValueError('Use the actual HTTPS repository page URL on GitHub or GitLab.')
    out=ROOT/'report';out.mkdir(exist_ok=True);r=ROOT/'results'
    env=json.loads((r/'environment.json').read_text());stats=json.loads((r/'dataset_stats.json').read_text());summary=pd.read_csv(r/'summary.csv');metrics=pd.read_csv(r/'metrics.csv');cmp=json.loads((r/'comparison.json').read_text());overlap=json.loads((r/'overlap_sensitivity.json').read_text())
    # Regenerate chart labels from saved measurements without retraining.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.2,3.4))
    for k,label in [('word','Word'),('char','Character'),('combined','Combined')]:
        s=summary[(summary.model==k)&(summary.budget!='pool')].copy();s['k']=s.budget.astype(int);s=s.sort_values('k')
        ax.errorbar(s.k,s.accuracy*100,yerr=s.accuracy_std*100,marker='o',capsize=4,label=label)
    ax.set(xlabel='Maximum training examples per class',ylabel='Test accuracy (%)',xticks=[10,30,60]);ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(r/'learning_curve.png',dpi=180);plt.close(fig)
    pdfmetrics.registerFont(TTFont('EmbeddedSerif', str(ROOT/'assets/DejaVuSerif.ttf')))
    pdfmetrics.registerFont(TTFont('EmbeddedSerifBold', str(ROOT/'assets/DejaVuSerif-Bold.ttf')))
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Body',fontName='EmbeddedSerif',fontSize=10,leading=12,spaceAfter=6))
    styles.add(ParagraphStyle(name='SmallBody',fontName='EmbeddedSerif',fontSize=8.5,leading=11,spaceAfter=5))
    styles.add(ParagraphStyle(name='TitleOwn',fontName='EmbeddedSerifBold',fontSize=18,leading=22,alignment=TA_CENTER,spaceAfter=12))
    styles['Heading1'].fontName='EmbeddedSerifBold';styles['Heading1'].fontSize=14;styles['Heading1'].leading=17
    styles['Heading2'].fontName='EmbeddedSerifBold';styles['Heading2'].fontSize=12
    story=[]; blocks=[]
    def p(text,small=False):story.append(Paragraph(text,styles['SmallBody' if small else 'Body']));blocks.append(('p',text))
    def h(text,sub=False):story.append(Paragraph(text,styles['Heading2' if sub else 'Heading1']));blocks.append(('h2' if sub else 'h1',text))
    def page():story.append(PageBreak());blocks.append(('page',''))
    def table(rows,widths):
        ts=[[Paragraph(html.escape(str(v)),styles['SmallBody']) for v in row] for row in rows]
        t=Table(ts,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9edf1')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.black),('LINEBELOW',(0,-1),(-1,-1),.4,colors.grey),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]));story.extend([t,Spacer(1,9)]);blocks.append(('table',rows))
    def result(model,budget='pool'):return summary[(summary.model==model)&(summary.budget==budget)].iloc[0]
    combined=result('combined');word=result('word');char=result('char')
    title='Banking Intent Classification with\nWord and Character TF-IDF'
    story.append(Paragraph(title.replace('\n','<br/>'),styles['TitleOwn']))
    p(html.escape(args.author)+' | NLP Course Final Project | September 2026')
    h('Abstract')
    p(f'We study a lightweight classifier for 77 banking intents using BANKING77. The method concatenates word and character TF-IDF vectors and trains a linear support vector machine. A controlled feature ablation evaluates three training-data caps over three random seeds, followed by a larger training-pool experiment. On the official test set, the combined model reaches {100*combined.accuracy:.2f}% accuracy and {100*combined.macro_f1:.2f}% macro-F1, compared with {100*word.accuracy:.2f}% accuracy for word features alone. The experiments examine how the benefit of combining the two feature types changes with the amount of training data.')
    if args.repo_url:p('Project code: <link href="'+html.escape(args.repo_url)+'">'+html.escape(args.repo_url)+'</link>.')
    else:p('Project code accompanies this report in banking77_project.zip. The repository URL is supplied separately in the course submission field.')
    h('1 Introduction')
    p('Banking support systems must route short user messages to the correct action. Closely related requests, such as a delayed card and an estimated delivery time, make this a fine-grained language classification problem. A linear classifier is a useful baseline here: it is fast to train and its features can be inspected directly.')
    p('This study tests whether character n-grams improve a word-based classifier when training examples are limited. Word n-grams retain local phrases; character n-grams share information across related spellings and word forms. The hypothesis is that these representations complement each other. All features are learned from BANKING77; no external training data are used.')
    h('1.1 Team',True)
    p(html.escape(args.author)+'. Individual project: data preparation, implementation, experiments and analysis.')
    h('2 Related Work')
    p('Wang and Manning [1] demonstrate that feature choice and linear-model variants can strongly affect text classification. Their NB-SVM uses supervised log-count-ratio features; our method uses ordinary TF-IDF and does not reproduce NB-SVM. Their findings motivate a careful lexical baseline, but sentiment classification scores are not directly comparable with banking-intent accuracy.')
    p('BERT [2] learns bidirectional contextual representations through pretraining, followed by task adaptation. Unlike our sparse model, it can represent a word differently across contexts. Sentence-BERT [3] adapts transformer encoders to obtain sentence representations through siamese/triplet training, enabling efficient embedding-based comparison. These are relevant semantic alternatives; we do not run them in this CPU-only study.')
    page()
    h('2 Related Work (continued)')
    p('Casanueva et al. [4] introduced BANKING77 and compared fixed sentence encoders with fine-tuned BERT. Their Table 3 reports 93.66% accuracy for BERT-TUNED and 93.36% for USE+ConveRT with full training data; the latter obtains 85.19% with 10 examples per intent. They provide reference points, although their training protocol differs from the validation split and sampling used here.')
    p('SetFit [5] adapts a sentence transformer using contrastive text pairs and then trains a classification head. Unlike TF-IDF, it starts with pretrained semantic representations. A comparison with SetFit would help separate the effect of feature choice from the benefit of pretraining; it is outside the experiments reported here.')
    h('3 Model Description')
    p('The main model has two parallel feature branches. The word branch extracts unigrams and bigrams, capped at 50,000 features. The character branch extracts within-word character n-grams of lengths 3 to 5, capped at 80,000 features. Lowercasing is enabled; stop-word removal, stemming and lemmatization are disabled. The word branch uses the scikit-learn default token pattern (tokens with at least two word characters). The character branch uses char_wb, including boundary padding.')
    p('Each branch uses sublinear term frequency and smoothed inverse document frequency. For a feature t appearing f(t,d) times in document d, tf = 1 + ln(f) for f &gt; 0 and zero otherwise. With N training documents and document frequency df(t), idf(t) = ln((1 + N)/(1 + df(t))) + 1. TF-IDF is the product of these quantities, followed by L2 normalization. Vocabulary and IDF are learned only from the current training subset.')
    p('Let w(d) and c(d) be the two normalized sparse vectors. The combined representation is x(d) = normalize([w(d); c(d)]). When both branches are nonzero, this gives each branch equal norm. A second normalization prevents the concatenated vector from simply having a larger norm than a single-branch baseline.')
    p('A one-vs-rest LinearSVC classifier learns 77 linear decision functions. For each binary problem it minimizes L2 regularization plus C times the squared hinge loss. Prediction is argmax over the class decision scores. We use C = 1, tolerance 0.0001, dual optimization and at most 5,000 iterations. Scores are margins, not calibrated probabilities. Removing either feature branch produces the ablation baselines, while all classifier settings remain identical.')
    h('4 Dataset')
    p('We use the public English BANKING77 dataset [4, 6], downloaded from the authors\' repository. Its license is CC BY 4.0; attribution and the license text are included with the data snapshot. The original intent labels are used without additional annotation.')
    table([['Data partition','Examples','Role'],['Official train',stats['official_train'],'Source of all training/validation data'],['Cleaned official train',stats['clean_train'],'After training-only deduplication'],['Training pool',stats['training_pool'],'Model fitting and nested subsampling'],['Validation',stats['validation'],'Held out; diagnostic evaluation only'],['Official test',stats['official_test'],'Fixed final comparison']], [150,70,275])
    p(f'Normalization for duplicate detection consists of lowercasing and whitespace collapsing. We remove {stats["removed_train_rows"]} duplicate training rows; no conflicting normalized texts are found in train. A fixed stratified split reserves 20% for validation. The training pool has 28 to 149 examples per class, while test contains 40 per class. Mean whitespace-token length is {stats["mean_train_words"]:.2f} in official train and {stats["mean_test_words"]:.2f} in test. Exact row indices and file SHA-256 hashes are saved.')
    page()
    h('5 Experiments')
    h('5.1 Metrics',True)
    p('Accuracy = (number of correct predictions) / N. For class k, F1(k) = 2 TP(k) / (2 TP(k) + FP(k) + FN(k)); macro-F1 is the arithmetic mean across all 77 classes. Accuracy is directly interpretable on the balanced official test set, while macro-F1 also reflects classwise precision/recall differences. All result tables express these metrics as percentages.')
    p('Reported standard deviations are sample standard deviations across training-subset seeds, not standard errors or confidence intervals. A paired bootstrap additionally resamples the 3,080 test rows 2,000 times and estimates the 2.5th/97.5th percentiles of the combined-minus-word accuracy difference. This interval is conditional on the fitted models and does not quantify training-set or domain-shift uncertainty.')
    h('5.2 Experiment Setup',True)
    p('The split seed is 2026. For each cap in {10, 30, 60}, examples within each class are shuffled with seeds {42, 43, 44}; the first min(cap, available) rows are retained. The same rows are used by all three methods, and subsets are nested across caps within a seed. Rare classes are not oversampled. These caps give 770, 2,308 and 4,537 training examples, respectively; therefore only the 10-example regime is exactly balanced.')
    p('A further experiment uses all 7,999 training-pool rows once, with seed 42. There are 30 model fits in total: 3 methods x 3 caps x 3 seeds, plus 3 pool fits. The pool condition is not the complete official training set. No hyperparameter grid search is performed: the model settings and comparison grid are fixed before test evaluation. Validation predictions are recorded for diagnostics only, and no settings are selected from test scores. Error analysis is performed after this fixed comparison.')
    p('The validation set requires an additional 2,000 labeled examples beyond each fitting budget, even though it is not used to tune the models. Consequently this is a study of fitting-data caps, not a claim of an end-to-end system needing only 10 labels per intent. This distinction matters when interpreting published few-shot results.')
    p(f'The measurements were executed with Python {env["python"].split()[0]} on CPU ({html.escape(env["cpu"])}), without a GPU. The environment exposes {env["logical_cpus"]} logical CPUs; OpenBLAS/OMP threads are limited to 2. Package versions are scikit-learn {env["packages"]["sklearn"]}, NumPy {env["packages"]["numpy"]}, pandas {env["packages"]["pandas"]} and SciPy {env["packages"]["scipy"]}. The Colab notebook reproduces the pipeline; timing depends on the runtime. Measured fit time includes feature fitting/transformation and classifier fitting, but excludes downloads, reporting and serialization.')
    h('5.3 Baselines',True)
    p('Word-only TF-IDF + LinearSVC tests whether character information is necessary. Character-only TF-IDF + LinearSVC tests whether word phrases add information beyond substrings. The combined model is the proposed practical configuration. A majority-class predictor provides a sanity check: its test accuracy is 1/77 = 1.30%, because the test set is balanced. The neural methods discussed in Related Work are used only as published reference results.')
    h('5.4 Reproducibility and leakage checks',True)
    p('Training and validation normalized text keys do not overlap. No test text is used to learn the vocabulary or IDF. Seven normalized test texts also occur in the cleaned official training source. We keep the official test unchanged for the main table and separately evaluate after removing these seven test rows. This diagnostic does not alter any fitted model. The project contains predictions for every run, split/subsample indices, per-class reports, data hashes, package versions and serialized pool models.')
    page()
    h('6 Results')
    rows=[['Fit cap / size','Model','Accuracy (%)','Macro-F1 (%)']]
    for budget in ['10','30','60','pool']:
        for kind in ['word','char','combined']:
            s=result(kind,budget);n=int(metrics[(metrics.budget==budget)&(metrics.model==kind)].iloc[0].n_train)
            lab=f'Up to {budget} / {n}' if budget!='pool' else f'Pool / {n}'
            acc=f'{100*s.accuracy:.2f}';f1=f'{100*s.macro_f1:.2f}'
            if budget!='pool':acc+=f' +/- {100*s.accuracy_std:.2f}';f1+=f' +/- {100*s.macro_f1_std:.2f}'
            rows.append([lab,kind,acc,f1])
    table(rows,[135,85,137,138])
    p('Table 1. Official test performance. Capped regimes show mean +/- sample standard deviation over three seeds. Pool rows are single fits; no between-seed uncertainty is estimated for them.',True)
    story.append(Image(str(r/'learning_curve.png'),width=17.1*cm,height=8.075*cm));blocks.append(('image','../results/learning_curve.png'))
    p('Figure 1. Learning curves with one-standard-deviation error bars. The x-axis is a per-class cap, not a guaranteed count for every class.',True)
    p(f'The combined representation has the highest mean accuracy at each tested cap and in the pool condition. At cap 10 its advantage over word features is {100*(result("combined","10").accuracy-result("word","10").accuracy):.2f} percentage points. The gain shrinks as more training data become available. This is consistent with complementary substring information being particularly useful when word statistics are sparse, but does not establish robustness to arbitrary spelling errors.')
    page()
    h('6.1 Comparison, cost and sensitivity',True)
    p(f'For the pool models, combined features improve accuracy by {100*cmp["delta_accuracy"]:.2f} percentage points over word features. The paired bootstrap 95% interval is [{100*cmp["bootstrap_ci95"][0]:.2f}, {100*cmp["bootstrap_ci95"][1]:.2f}] percentage points. The interval is unadjusted and applies to this pair of fitted models.')
    p(f'Pool training takes {word.fit_seconds:.2f} s for word features, {char.fit_seconds:.2f} s for character features and {combined.fit_seconds:.2f} s for the combination on the measured machine. These are single-run timings, not a rigorous latency benchmark. Removing the seven overlapping test rows leaves {overlap["n_test_nonoverlap"]} rows and gives {100*overlap["combined"]:.2f}% combined accuracy; the main conclusion is unchanged in this diagnostic.')
    table([['Method','Training protocol','Accuracy (%)'],['Word + character SVM (ours)','7,999 training-pool examples',f'{100*combined.accuracy:.2f}'],['BERT-TUNED [4]','Full official train; published', '93.66'],['USE + ConveRT [4]','Full official train; published','93.36'],['USE + ConveRT [4]','10 per intent; published','85.19']], [180,215,100])
    p('Table 2. Historical context from [4], Table 3. Neural methods were not rerun. Training budgets, subsets, pretraining and validation protocols differ; these numbers do not constitute a controlled ranking. The combined classifier remains below both full-training neural reference scores.',True)
    h('6.2 Error Analysis',True)
    p('The combined pool model makes 280 errors. Frequent directional confusions include card arrival versus delivery estimate (4 cases), fiat-currency support versus supported cards/currencies (4), and failed versus reverted top-up (4). Verification-related classes are also confused. These distinctions often depend on the purpose of the request rather than isolated keywords.')
    table([['Test message','True intent','Prediction'],['When will I get my card?','card arrival','card delivery estimate'],['Can I link another card to my account?','card linking','getting spare card'],['Is it a good time to exchange?','exchange rate','exchange charge'],['There is a fee I do not recognize on my statement.','extra charge on statement','card payment not recognised']], [235,130,130])
    p('Table 3. Actual erroneous predictions; the final message is lightly paraphrased for readability. The full original text and all 280 errors are in results/errors_combined.csv. The examples show where lexical overlap and missing transaction context make the category distinction difficult.',True)
    p('A lexical representation cannot reliably resolve paraphrases with little shared vocabulary, long-range composition or implicit context. Character features can also amplify irrelevant surface similarity. The error review is qualitative. Separate tests on paraphrases, spelling noise and out-of-domain requests are needed to assess generalization.')
    page()
    h('7 Conclusion')
    p(f'The experiments show that a linear classifier can benefit from combining word and character features for banking-intent classification. Combining word and character TF-IDF with a linear SVM reaches {100*combined.accuracy:.2f}% official-test accuracy and outperforms the two single-branch baselines in the tested regimes. The learning curves show a larger benefit over word features when fitting data are scarce. Code, data provenance, predictions and uncertainty diagnostics accompany the report.')
    p('The conclusions are limited to one English-language dataset. The capped experiments use three sampling seeds, while the pool condition has a single fit. The next step is to compare the same splits with a frozen sentence encoder and SetFit, then test the models on paraphrases and requests outside the banking domain.')
    h('References')
    refs=[
    ('1','Sida Wang and Christopher D. Manning. 2012. Baselines and Bigrams: Simple, Good Sentiment and Topic Classification. ACL, pp. 90-94.','https://aclanthology.org/P12-2018/'),
    ('2','Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT, pp. 4171-4186.','https://aclanthology.org/N19-1423/'),
    ('3','Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP, pp. 3982-3992.','https://aclanthology.org/D19-1410/'),
    ('4','Inigo Casanueva, Tadas Temcinas, Daniela Gerz, Matthew Henderson and Ivan Vulic. 2020. Efficient Intent Detection with Dual Sentence Encoders. NLP for Conversational AI, pp. 38-45.','https://aclanthology.org/2020.nlp4convai-1.5/'),
    ('5','Lewis Tunstall et al. 2022. Efficient Few-Shot Learning Without Prompts. arXiv:2209.11055.','https://arxiv.org/abs/2209.11055'),
    ('6','PolyAI. Task-specific datasets: BANKING77 data and CC BY 4.0 license. Data snapshot retrieved 20 September 2026.','https://github.com/PolyAI-LDN/task-specific-datasets'),
    ]
    for num,text,url in refs:p(f'[{num}] {html.escape(text)} <link href="{url}" color="#234e70">{url}</link>',True)
    def footer(canvas,doc):
        canvas.setFont('EmbeddedSerif',9);canvas.drawCentredString(297.6,28,str(doc.page))
    doc=SimpleDocTemplate(str(out/'report.pdf'),pagesize=(595.28,841.89),rightMargin=50,leftMargin=50,topMargin=42,bottomMargin=42,title=title.replace('\n',' '),author=args.author)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    # Editable LaTeX with exactly the same section order/content; article layout.
    def tex(s):
        s=re.sub(r'<[^>]+>','',s);s=html.unescape(s)
        return ''.join({'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}.get(c,c) for c in s)
    parts=[r'\documentclass{article}',r'\usepackage[utf8]{inputenc}',r'\usepackage[T1]{fontenc}',r'\usepackage{amsmath,amssymb,graphicx,hyperref,longtable}',r'\usepackage[margin=2.4cm]{geometry}',r'\title{'+tex(title.replace('\n',' '))+'}',r'\author{'+tex(args.author)+'}',r'\date{September 2026}',r'\begin{document}\maketitle']
    for kind,val in blocks:
        if kind=='p':parts.append(tex(val)+'\n')
        elif kind in ['h1','h2']:parts.append(('\\section*{' if kind=='h1' else '\\subsection*{')+tex(val)+'}')
        elif kind=='page':parts.append(r'\clearpage')
        elif kind=='image':parts.append(r'\begin{center}\includegraphics[width=\linewidth]{'+val+r'}\end{center}')
        elif kind=='table':
            cols=len(val[0]);w=round(14.5/cols,2);parts.append(r'\begin{longtable}{'+''.join('p{'+str(w)+'cm}' for _ in range(cols))+'}')
            parts.extend(' & '.join(tex(str(v)) for v in row)+r' \\' for row in val);parts.append(r'\end{longtable}')
    parts.append(r'\end{document}');(out/'main.tex').write_text('\n'.join(parts))
    (out/'metadata.json').write_text(json.dumps({'author':args.author,'repo_url':args.repo_url},indent=2))
    print(out/'report.pdf')
if __name__=='__main__':main()
