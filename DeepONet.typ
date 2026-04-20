#import "@preview/polylux:0.4.0": *
#import "@preview/touying:0.6.1"
#import "touying-template/theme.typ": *
#import "ori.typ": *
#import "@preview/cetz:0.4.2"
#import "@preview/wrap-it:0.1.1": *

#show: mytheme.with(
  aspect-ratio: "16-9",
  font-en: "Noto Sans",
  font-ja: "BIZ UDPGothic",
  font-math: "Noto Sans Math",
  config-info(
    title: [],
    subtitle: [Subtitle],
    author: [Author],
    institution: [Institution],
    header: [Conference\@Location (Date)]
  )
)

= 数据准备
