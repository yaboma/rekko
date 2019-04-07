
def average_precision(
        dict data_true,
        dict data_predicted,
        const unsigned long int k
) -> float:
    cdef:
        unsigned long int n_items_predicted
        unsigned long int n_items_true
        unsigned long int n_correct_items
        unsigned long int item_idx

        double average_precision_sum
        double precision

        set items_true
        list items_predicted

    if not data_true:
        raise ValueError('data_true is empty')

    average_precision_sum = 0.0

    for key, items_true in data_true.items():
        items_predicted = data_predicted.get(key, [])

        n_items_true = len(items_true)
        n_items_predicted = min(len(items_predicted), k)

        if n_items_true == 0 or n_items_predicted == 0:
            continue

        n_correct_items = 0
        precision = 0.0

        for item_idx in range(n_items_predicted):
            if items_predicted[item_idx] in items_true:
                n_correct_items += 1
                precision += <double>n_correct_items / <double>(item_idx + 1)

        average_precision_sum += <double>precision / <double>min(n_items_true, k)

    return average_precision_sum / <double>len(data_true)