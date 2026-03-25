\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{phys400proposal}[2026 Project Proposal Class]

\LoadClass[12pt,a4paper]{article}

% ------------------------
% Packages
% ------------------------
\RequirePackage{geometry}
\RequirePackage{setspace}
\RequirePackage{graphicx}
\RequirePackage{amsmath,amssymb}
\RequirePackage{natbib}
\RequirePackage{hyperref}
\RequirePackage{fancyhdr}
\RequirePackage[table]{xcolor}
\RequirePackage{array}
\RequirePackage{titlesec}
\RequirePackage{tcolorbox}
\RequirePackage{pgfgantt}

% ------------------------
% Layout
% ------------------------
\geometry{left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm}
\onehalfspacing
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\definecolor{tealblue}{RGB}{0,120,130}
\definecolor{lightteal}{RGB}{220,245,246}
% ------------------------
% Header
% ------------------------
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{PHYS400 Project Proposal}
\fancyhead[R]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}

% ------------------------
% Gray Section Box
% ------------------------
\newcommand{\graysectionbox}[1]{%
  %\vspace{12pt}
  \noindent
  \colorbox[gray]{0.85}{%
    \parbox{\dimexpr\textwidth-2\fboxsep}{%
      \centering\large\bfseries #1%
    }%
  }%
  %\vspace{8pt}
}

\titleformat{\section}{\normalfont}{}{0pt}{\graysectionbox}
\titlespacing*{\section}{0pt}{24pt}{12pt}

% ------------------------
% 0.7 Textwidth Boxed Abstract
% ------------------------
\newenvironment{boxedabstract}{
    \begin{center}
        \begin{minipage}{0.8\textwidth}
            \section*{Abstract}
            \addcontentsline{toc}{section}{Abstract}
            \begin{tcolorbox}[
                colback=gray!10,
                colframe=black,
                width=\textwidth,
                boxrule=0.8pt,
                arc=0pt
            ]
}{
            \end{tcolorbox}
        \end{minipage}
    \end{center}
    \vspace{12pt}
}

% ------------------------
% Information Fields
% ------------------------
\newcommand{\proposaltitle}[1]{\def\@proposaltitle{#1}}
\newcommand{\studentname}[1]{\def\@studentname{#1}}
\newcommand{\studentid}[1]{\def\@studentid{#1}}
\newcommand{\advisor}[1]{\def\@advisor{#1}}
\newcommand{\studyfield}[1]{\def\@studyfield{#1}}
\newcommand{\studytype}[1]{\def\@studytype{#1}}
\newcommand{\submissiondate}[1]{\def\@submissiondate{#1}}

% ------------------------
% Original Tabular Cover Page (Gray Left Cells Only)
% ------------------------
\newcommand{\makeproposalcover}{
\begin{titlepage}
\centering
\vspace*{1cm}
{\Large \textcolor{black}{\textbf{PHYS400 \\ SPECIAL PROBLEMS IN PHYSICS \\ PROJECT PROPOSAL}} \par}
\vspace{1.5cm}
\renewcommand{\arraystretch}{1.6}
\begin{tabular}{|>{\columncolor[gray]{0.9}}p{5cm}|p{8cm}|}
\hline
Project Title & \@proposaltitle \\ \hline
Student Name & \@studentname \\ \hline
Student ID & \@studentid \\ \hline
Advisor & \@advisor \\ \hline
Field of study & \@studyfield \\ \hline
Type of study & \@studytype  \\ \hline
Submission Date & \@submissiondate \\ \hline

\end{tabular}
\vfill
\end{titlepage}
}

% ------------------------
% Modern 15-Week Semester Gantt
% ------------------------
\newenvironment{semestergantt}{
    \section{Timeline}
    \begin{ganttchart}[
        y unit title=0.6cm,
        y unit chart=0.6cm,
        vgrid={},
        hgrid,
        title height=1,
        bar/.style={fill=blue!60},
        bar height=0.6
    ]{1}{15}
    \gantttitle{Semester Plan (15 Weeks)}{15} \\
    \gantttitlelist{1,...,15}{1} \\
}{
    \end{ganttchart}
}

\newcommand{\workpackage}[3]{
    \ganttbar{#1}{#2}{#3} \\
}

% ------------------------
% Sections
% ------------------------
\newcommand{\introduction}{\section{Introduction}}
\newcommand{\literaturereview}{\section{Literature Review}}
\newcommand{\researchquestion}{\section{Research Question}}
\newcommand{\methodology}{\section{Methodology}}
\newcommand{\expectedresults}{\section{Expected Results}}

\newcommand{\referencessection}{
\bibliographystyle{apalike}
\bibliography{references}
}
