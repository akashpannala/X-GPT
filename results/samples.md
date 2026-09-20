# Sample outputs (verbatim, checkpoint.pt @ iter 4999, val 1.3799)

## 1. `python sample.py --prompt "Kural 1" --max_tokens 300 --seed 7`

```
Kural 1112
Tamil: இருவின் பக்க பொழுகலம் அறிந்தர்க்கு அந்தொழுகும் மற்றின் பொய்தக்க ஆற்றறம் தருந்தார்.
Transliteration: Elino Thavameentu Ayirkku Eniyang Kaalar Patavi Aripakku Irindhadhu Pol
English: Those who who would proverty their who is in self of his who are fores his excetsised firmnds will with says
```

Honest note: structure (Tamil + transliteration + English) is learned,
Tamil lines scan okay-ish, English is still broken. Needs more iters/data.

## 2. `python sample.py --prompt "Tamil:" --max_tokens 200 --seed 7`

```
Tamil: மீற்றிய அஃதிற்பிறந்து உன்றிசல் சுற்றாண்மை இரப்பது இல்.
Transliteration: Payarai Ondraththain Nananjal Seyalam Elindhu Arabidhu Kayidhu
English: Though at the good in they seafhen, who has ven to him
```

Same story: format holds, short Tamil couplets look plausible,
English trails off. Next step would be longer training or cleaner data.
