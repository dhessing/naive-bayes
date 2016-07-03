(ns naive-bayes.core
  (:use clojure.test))

(defn parse [observations]
  (vec (for [class (map second (group-by first observations))]
         (into {} (map (fn [[k v]] [k (frequencies v)]) (seq (apply merge-with concat class)))))))

(defn classes [data]
  (map (fn [class] [(ffirst class) ((comp ffirst second first) class)]) data))

(defn p [data feature value]
  (/ (reduce + (keep #(% value) (keep feature data)))
     (reduce + (mapcat vals (keep feature data)))))

(defn p-given-class [data feature value class-key class-value]
  (let [class (first (filter #(get-in % [class-key class-value]) data))]
    (/ (get-in class [feature value] 0)
       (reduce + (vals (feature class))))))

(defn p-given-feature [data class-key class-value feature value]
  (let [class (first (filter #(get-in % [class-key class-value]) data))]
    (/ (get-in class [feature value])
       (reduce + (keep #(get-in % [feature value]) data)))))

(defn naive-bayes [data class-key class-value & events]
  (let [events (partition 2 events)]
    (/ (* (p data class-key class-value)
          (reduce * (map (fn [[feature value]] (p-given-class data feature value class-key class-value)) events)))
       (reduce + (map (fn [[class-key class-value]]
                        (* (p data class-key class-value)
                           (reduce * (map (fn [[feature value]] (p-given-class data feature value class-key class-value)) events))))
                      (classes data))))))

(defn classify [data & events]
  (let [events (partition 2 events)]
    (apply max-key (fn [[class-key class-value]]
                     (* (p data class-key class-value)
                        (reduce * (map (fn [[feature value]] (p-given-class data feature value class-key class-value)) events))))
           (classes data))))

