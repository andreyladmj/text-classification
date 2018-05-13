import sys

import pymysql
from Cython.Shadow import profile
from nltk import word_tokenize

local_db_connection = pymysql.connect(host='localhost', database='edusson', user='root', password='')
db_cursor_read = local_db_connection.cursor()
db_cursor_write = local_db_connection.cursor()


def preprocess():
    db_cursor_read.execute('SELECT * FROM es_orders')
    c = 0
    for row in db_cursor_read:
        c += 1
        row = prepare_sql(row)
        insert_into_prepared_table(row)
        print('\rProcessed: {} rows'.format(c), end='')

    print('\nTotal:', c)
    local_db_connection.commit()


def prepare_sql(row):
    ID, SITE_ID, CUSTOMER_ID, WRITER_ID, STATE_ID, currency_id, is_paid_order, \
    order_total_usd, estimated_total, order_date, date_started, date_finished, \
    deadline, name, description, is_first_client_order, is_easy_bidding = range(17)

    return dict(
        id = row[ID],
        site_id = row[SITE_ID],
        customer_id = row[CUSTOMER_ID],
        writer_id = row[WRITER_ID],
        state_id = row[STATE_ID],
        currency_id = row[currency_id],
        is_paid_order = row[is_paid_order],
        order_total_usd = row[order_total_usd],
        estimated_total = row[estimated_total],
        order_date = row[order_date],
        date_started = row[date_started],
        date_finished = row[date_finished],
        deadline = row[deadline],
        name = prepare_text(row[name]),
        description = prepare_text(row[description]),
        is_first_client_order = row[is_first_client_order],
        is_easy_bidding = row[is_easy_bidding],
    )


def prepare_text(text):
    if not text: return ''

    text = text.lower().replace('\\', '')
    words = word_tokenize(text)
    return words


def insert_into_prepared_table(data):
    diff = (data['deadline'] - data['order_date'])

    row = dict(
        order_id=data['id'],
        state_id=data['state_id'],
        order_total_usd=data.get('order_total_usd', '') or 0,
        estimated_total=data.get('estimated_total', '') or 0,
        deadline=diff.total_seconds(),
        name=' '.join(data['name']),
        description=' '.join(data['description']),
        is_first_client_order=data.get('is_first_client_order', '') or 0,
        name_words=len(data['name']),
        description_words=len(data['description']),
    )

    sql = 'INSERT INTO prepared_orders VALUES({order_id}, {state_id}, {order_total_usd},' \
          '{estimated_total}, "{deadline}", "{name}", "{description}", {is_first_client_order}, ' \
          '{name_words}, {description_words});'
    query = sql.format(**row)

    try:
        db_cursor_write.execute(query)
    except Exception as e:
        print(query)
        raise e


def days_hours_minutes(td):
    return td.days, td.seconds//3600, (td.seconds//60)%60


# '1', 'Bidding', NULL, '1'
# '2', 'In Progress', NULL, '1'
# '3', 'Finished', NULL, '1'
# '4', 'Canceled by Customer', NULL, '1'
# '5', 'Canceled by Writer', NULL, '1'
# '6', 'Canceled by System', NULL, '1'
# '9', 'Not Active', NULL, '2'
# '10', 'Active', NULL, '2'
# '11', 'Expired', NULL, '2'
# '12', 'Pending Payment', NULL, '1'
# '13', 'Under Investigation', NULL, '1'
# '14', 'Pending Writer', NULL, '1'


if __name__ == '__main__':
    preprocess()