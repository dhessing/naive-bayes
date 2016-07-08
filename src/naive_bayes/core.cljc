(ns naive-bayes.core)

(defn bag-of-words [observations]
  (vec (for [class (map second (group-by first observations))]
         (into {} (map (fn [[k v]] [k (frequencies v)]) (seq (apply merge-with concat class)))))))

(defn classes [data]
  (map (fn [class] [(ffirst class) ((comp ffirst second first) class)]) data))

(defn p [data k feature value]
  (/ (reduce + k (keep #(% value) (keep feature data)))
     (reduce + (* k (count (classes data))) (mapcat vals (keep feature data)))))

(defn p-given-class [data k feature value class-key class-value]
  (let [class (first (filter #(get-in % [class-key class-value]) data))]
    (/ (+ k (get-in class [feature value] 0))
       (reduce + (* k (count (set (mapcat keys (keep feature data))))) (vals (feature class))))))

(defn p-given-feature [data class-key class-value feature value]
  (let [class (first (filter #(get-in % [class-key class-value]) data))]
    (/ (get-in class [feature value])
       (reduce + (keep #(get-in % [feature value]) data)))))

(defn prior-times-likelihood [data k class-key class-value events]
  (* (p data k class-key class-value)
     (reduce * (map (fn [[feature value]] (p-given-class data k feature value class-key class-value)) events))))

(defn naive-bayes [data k class-key class-value & events]
  (let [events (partition 2 events)]
    (/ (prior-times-likelihood data k class-key class-value events)
       (reduce + (map (fn [[class-key class-value]]
                        (prior-times-likelihood data k class-key class-value events))
                      (classes data))))))

(defn classify
  ([data events]
    (classify data 1 events))
  ([data k events]
   (let [events (partition 2 events)]
     (apply max-key
            (fn [[class-key class-value]]
              (prior-times-likelihood data k class-key class-value events))
            (classes data)))))

(defn string->words [string]
  (-> string
      (clojure.string/lower-case)
      (clojure.string/replace #"[^a-z\s]" "")
      (clojure.string/replace #"\s\s+" " ")
      (clojure.string/split #"\s")))

(defn classify-text
  ([data attribute text]
   (classify-text data 1 attribute text))
  ([data laplace-smoothing attribute text]
   (classify data laplace-smoothing (mapcat (fn [word] [attribute word]) (string->words text)))))

