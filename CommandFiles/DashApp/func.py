from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn import svm
import pandas as pd
import copy
from ordinal_classifier import OrdinalClassifier
import re, string


class Classifier:

    def __init__(self, clf, text_type, df, use_ord=False):
        self.clf = clf
        self.text_type = text_type
        self.use_ord = use_ord
        self.df = df
        self.emotions = ['Sentyment_nom', 'Lubienie', 'Wstręt', 'Złość', 'Strach', 'Zaskoczenie', 'Oczekiwanie', 'Radość',
                         'Smutek']
        self.emo_clfs = {}
        if text_type == 1:
            self.text = 'CleanedText'
        elif text_type == 2:
            self.text = 'RemovedSW'

    def prepare_new_text(self, text):

        text = text.lower()
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('[‘’“”…]', '', text)
        text = text.replace(' –', '')
        text = re.sub('\n', ' ', text)
        if self.text_type == 2:
            stop_words = ['a', 'aby', 'ach', 'acz', 'aczkolwiek', 'aj', 'albo', 'ale', 'alez', 'ależ', 'ani', 'az',
                          'aż', 'bardziej',
                          'bardzo', 'beda', 'bedzie', 'bez', 'deda', 'będą', 'bede', 'będę', 'będzie', 'bo', 'bowiem',
                          'by', 'byc',
                          'być', 'byl', 'byla', 'byli', 'bylo', 'byly', 'był', 'była', 'było', 'były', 'bynajmniej',
                          'cala', 'cali',
                          'caly', 'cała', 'cały', 'ci', 'cie', 'ciebie', 'cię', 'co', 'cokolwiek', 'cos', 'coś',
                          'czasami', 'czasem',
                          'czemu', 'czy', 'czyli', 'daleko', 'dla', 'dlaczego', 'dlatego', 'do', 'dobrze', 'dokad',
                          'dokąd', 'dosc',
                          'dość', 'duzo', 'dużo', 'dwa', 'dwaj', 'dwie', 'dwoje', 'dzis', 'dzisiaj', 'dziś', 'gdy',
                          'gdyby', 'gdyz',
                          'gdyż', 'gdzie', 'gdziekolwiek', 'gdzies', 'gdzieś', 'go', 'i', 'ich', 'ile', 'im', 'inna',
                          'inne', 'inny',
                          'innych', 'iz', 'iż', 'ja', 'jak', 'jakas', 'jakaś', 'jakby', 'jaki', 'jakichs', 'jakichś',
                          'jakie', 'jakis',
                          'jakiś', 'jakiz', 'jakiż', 'jakkolwiek', 'jako', 'jakos', 'jakoś', 'ją', 'je', 'jeden',
                          'jedna', 'jednak',
                          'jednakze', 'jednakże', 'jedno', 'jego', 'jej', 'jemu', 'jesli', 'jest', 'jestem', 'jeszcze',
                          'jeśli',
                          'jezeli', 'jeżeli', 'juz', 'już', 'kazdy', 'każdy', 'kiedy', 'kilka', 'kims', 'kimś', 'kto',
                          'ktokolwiek',
                          'ktora', 'ktore', 'ktorego', 'ktorej', 'ktory', 'ktorych', 'ktorym', 'ktorzy', 'ktos', 'ktoś',
                          'która',
                          'które', 'którego', 'której', 'który', 'których', 'którym', 'którzy', 'ku', 'lat', 'lecz',
                          'lub', 'ma', 'mają',
                          'mało', 'mam', 'mi', 'miedzy', 'między', 'mimo', 'mna', 'mną', 'mnie', 'moga', 'mogą', 'moi',
                          'moim', 'moj',
                          'moja', 'moje', 'moze', 'mozliwe', 'mozna', 'może', 'możliwe', 'można', 'mój', 'mu', 'musi',
                          'my', 'na', 'nad',
                          'nam', 'nami', 'nas', 'nasi', 'nasz', 'nasza', 'nasze', 'naszego', 'naszych', 'natomiast',
                          'natychmiast',
                          'nawet', 'nia', 'nią', 'nic', 'nich', 'nie', 'niech', 'niego', 'niej', 'niemu', 'nigdy',
                          'nim', 'nimi', 'niz',
                          'niż', 'no', 'o', 'obok', 'od', 'około', 'on', 'ona', 'one', 'oni', 'ono', 'oraz', 'oto',
                          'owszem', 'pan',
                          'pana', 'pani', 'po', 'pod', 'podczas', 'pomimo', 'ponad', 'poniewaz', 'ponieważ', 'powinien',
                          'powinna',
                          'powinni', 'powinno', 'poza', 'prawie', 'przeciez', 'przecież', 'przed', 'przede', 'przedtem',
                          'przez', 'przy',
                          'roku', 'rowniez', 'również', 'sam', 'sama', 'są', 'sie', 'się', 'skad', 'skąd', 'soba',
                          'sobą', 'sobie',
                          'sposob', 'sposób', 'swoje', 'ta', 'tak', 'taka', 'taki', 'takie', 'takze', 'także', 'tam',
                          'te', 'tego',
                          'tej', 'ten', 'teraz', 'też', 'to', 'toba', 'tobą', 'tobie', 'totez', 'toteż', 'totobą',
                          'trzeba', 'tu',
                          'tutaj', 'twoi', 'twoim', 'twoj', 'twoja', 'twoje', 'twój', 'twym', 'ty', 'tych', 'tylko',
                          'tym', 'u', 'w',
                          'wam', 'wami', 'was', 'wasz', 'wasza', 'wasze', 'we', 'według', 'wiele', 'wielu', 'więc',
                          'więcej', 'wlasnie',
                          'właśnie', 'wszyscy', 'wszystkich', 'wszystkie', 'wszystkim', 'wszystko', 'wtedy', 'wy', 'z',
                          'za', 'zaden',
                          'zadna', 'zadne', 'zadnych', 'zapewne', 'zawsze', 'ze', 'zeby', 'zeznowu', 'zł', 'znow',
                          'znowu', 'znów',
                          'zostal', 'został', 'żaden', 'żadna', 'żadne', 'żadnych', 'że', 'żeby']
            text = text.split()
            text = [word for word in text if word not in stop_words]
            text = ' '.join(text)

        return text

    def text2vec(self):
        self.vector = CountVectorizer()
        X = self.df[self.text]
        X_vec = self.vector.fit_transform(X).toarray()
        return X_vec

    def predict_values_vec(self):
        val_dict = {}
        X_vec = self.text2vec()
        for emo in self.emotions:
            ylabels = self.df[emo]
            X_train, X_test, y_train, y_test = train_test_split(X_vec, ylabels, test_size=0.3)
            if self.use_ord and emo != 'Sentyment_nom':
                ord_clf = OrdinalClassifier(self.clf)
                ord_clf.fit(X_train, y_train)
                self.emo_clfs[emo] = copy.deepcopy(ord_clf)
                y_pred = self.emo_clfs[emo].predict(X_test)

            else:
                self.emo_clfs[emo] = copy.deepcopy(self.clf.fit(X_train, y_train))
                y_pred = self.emo_clfs[emo].predict(X_test)

            val_dict[emo] = (y_pred, y_test)

        return val_dict

    def predict_single_val(self, new_text):
        new_text = self.prepare_new_text(new_text)
        text_vec = self.vector.transform([new_text]).toarray()
        pred_dict = {}

        for emo in self.emotions:
            pred_dict[emo] = self.emo_clfs[emo].predict(text_vec)
        return pred_dict

    def calc_acc(self):
        acc_dict = {}
        prec_dict = {}
        rec_dict = {}
        f1_dict = {}
        conf_matrix_dict = {}
        val_dict = self.predict_values_vec()
        for k in val_dict.keys():
            y_pred, y_true = val_dict[k]
            acc_dict[k] = accuracy_score(y_true, y_pred)
            prec_dict[k] = precision_score(y_true, y_pred, average='macro', zero_division=0)
            rec_dict[k] = recall_score(y_true, y_pred, average='macro')
            f1_dict[k] = f1_score(y_true, y_pred, average='macro')
            conf_matrix_dict[k] = confusion_matrix(y_true, y_pred)
        return acc_dict, prec_dict, rec_dict, f1_dict, conf_matrix_dict
