-- Ahmad Arsyel 1301164193
--- Prepare data
local TRAIN_PATH = "DataTrain_Tugas3_AI.csv"
local TEST_PATH = "DataTest_Tugas3_AI.csv"
local ANSWER_PATH = "D:/TelkomUniversity/AI/Tugas_KNN/TebakanTugas3.csv"
--local ANSWER_PATH = "/media/reydvires/0AE8CA9CE8CA8603/TelkomUniversity/AI/" ..
  --  "Tugas_KNN/Alternative/TebakanTugas3.csv"

-- @param table Print traverse of table
local function print_table(table)
  for _, v in ipairs(table) do -- show parsing from CSV file
    if not v.y then -- case of label is ? (none)
      v.y = "?"
    end
    print(v.index, v.x1, v.x2, v.x3, v.x4, v.x5, v.y)
  end
end

-- @param path Use your path for .csv file
local function parse_CSV(path)
  local table_list = {}
  -- output will saved in table_list
  for line in io.lines(path) do
    local col1, col2, col3, col4, col5, col6, col7 = line:match(
        "%i*(.-),%s*(.-),%s*(.*),%s*(.*),%s*(.*),%s*(.*),%s*(.*)") -- converting
    table_list[#table_list + 1] = {
      index = col1,
      x1 = tonumber(col2),
      x2 = tonumber(col3),
      x3 = tonumber(col4),
      x4 = tonumber(col5),
      x5 = tonumber(col6),
      y = tonumber(col7)
    }
  end
  table.remove(table_list, 1) -- remove the title/header
  return table_list
end

-- @param path Use your own path for .csv raw file targeted
-- @param data_table Saving file to .csv from data_table
-- @param sep Separator of file
local function table_to_CSV(path, data_table, sep)
  sep = sep or ','
  local file = assert(io.open(path, "w")) -- w mean write
  for _, v in ipairs(data_table) do
    --print(v)
    file:write(v) -- v.y can be replaced
    file:write('\n')
  end
  file:close()
  print("file saved to CSV file\n")
end
---
-- fungsi menghitung jarak
local function euclidean(p_tab, q_tab)
  local sum = (p_tab.x1 - q_tab.x1)^2 + (p_tab.x2 - q_tab.x2)^2 +
      (p_tab.x3 - q_tab.x3)^2 + (p_tab.x4 - q_tab.x4)^2 +
      (p_tab.x5 - q_tab.x5)^2
  return math.sqrt(sum)
end

local function insertion_sort(tab)
  for i=2, #tab do
    local x = tab[i]
    local j = i-1
    while (j > 0) and (tab[j].range > x.range) do
      tab[j+1] = tab[j]
      j = j - 1
    end
    tab[j+1] = x
  end
end

local function copy_table(from_tab, to_tab)
  for _,v in ipairs(from_tab) do
    table.insert(to_tab, {
      index = v.index,
      x1 = v.x1,
      x2 = v.x2,
      x3 = v.x3,
      x4 = v.x4,
      x5 = v.x5,
      y = v.y
    })
  end
end

local function clear_t()
  return {
    test = {},
    validation = {}
  }
end

-- melakukan split data sebanyak k, dari metode k-fold cross validation
local function get_validation_list(train_tab, k)
  local temp_tab = {}
  local random_tab = {}
  local temp_rand_tab = {}

  -- copy data from train_tab
  copy_table(train_tab, temp_tab)
  -- do random placing in data train
  while #temp_tab > 0 do
    local a = math.random(1, #temp_tab)
    table.insert(random_tab, table.remove(temp_tab, a))
  end
  -- copy random_tab value to temp_rand_tab
  copy_table(random_tab, temp_rand_tab)

  temp_tab = {} -- do reset after insert
  t = clear_t()
  -- do split k data from train_list
  local sum_data = 0
  while #temp_tab < k do
    while #t.validation + #t.test < #train_list and sum_data < #train_list do
      if #t.validation < #train_list/k then
        table.insert(t.validation, table.remove(random_tab, 1))
      else
        table.insert(t.test, table.remove(random_tab, 1))
      end
      sum_data = sum_data + 1
    end
    table.insert(temp_tab, t)
    -- terbagi 2, sekarang train dimasuki validation set
    local new_validation = {}
    local new_train = {}
    for _,v in ipairs(t.validation) do
      table.insert(new_validation, v)
    end
    --print("copy tab validation",#new_validation)
    for _,v in ipairs(t.test) do
      table.insert(new_train, v)
    end
    --print("copy tab test",#new_train)
    while #new_validation > 0 do
      table.insert(new_train, table.remove(new_validation, 1))
    end
    -- do reset counter
    t = clear_t()
    sum_data = 0
    random_tab = new_train
  end
  return temp_tab
end

local function get_k_neighbor(tab, k)
  local knearest_tab = {}
  local i = 1
  while #knearest_tab < k do
    table.insert(knearest_tab, tab[i])
    i = i + 1
  end
  return knearest_tab
end

-- counting the most label appear in nearest neighbor
local function generate_new_label(tab)
  local label_selection = {0, 0, 0, 0}
  for _,v in ipairs(tab) do
    --print(v.range, v.label)
    if v.label == 0 then
      label_selection[1] = label_selection[1] + 1
    elseif v.label == 1 then
      label_selection[2] = label_selection[2] + 1
    elseif v.label == 2 then
      label_selection[3] = label_selection[3] + 1
    elseif v.label == 3 then
      label_selection[4] = label_selection[4] + 1
    end
  end

  local new_label = 0
  local max = label_selection[1]
  for i=2,#label_selection do
    if label_selection[i] > max then
      max = label_selection[i]
      new_label = i - 1
    end
  end
  return new_label
end

local function generate_k_from_validation(tab)
  local best_k = tab[1]
  for i=2,#tab do
    if tab[i].average >  best_k.average then -- base on average
      best_k = tab[i]
    end
  end
  return best_k
end

-- mencari best accuracy dari x data yang di generate
local function best_accuracy(tab)
  local ia = 1
  local b_ia = tab[ia]
  for i=2,#tab do
    if tab[i].accuracy > b_ia.accuracy then
      b_ia.accuracy = tab[i].accuracy
      b_ia.fold = tab[i].accuracy
      b_ia.irdata = tab[i].irdata
      b_ia.average = tab[i].average
    end
  end
  return b_ia
end

-- MAIN PROGRAM
math.randomseed(os.time())

local k_fold = 8
local generate_data = 1000 -- testing x data
local max_range = 7 -- try k-nearest = {5, 9, 13, 17, 21, 25, 29}

local k_select = {}
local validation_list = {}
local less_accuracy = {}
train_list = parse_CSV(TRAIN_PATH)
local test_list = parse_CSV(TEST_PATH)
local count_k = 1
local knn -- analize in 4*range+1, multiply of 4 +1
local range = 1
local average = 0

print("knn is on running...\ngenerate " ..
    generate_data.." new random data train base index position..." ..
    "\nrange of iterate " .. max_range
)

for i=1,generate_data do -- do generate x sample data
  validation_list[i] = get_validation_list(train_list, k_fold)
end

-- do validation
while (range <= max_range) do
  knn = range * 4 + 1
  for j=1,#validation_list do
    local result_tab = {}
    local new_label = {}
    local count_fold = 1
    local count_cross = 1
    local count_label = 0 -- true is_label_similar
    local is_label_similar = true
    local best_folds = {} -- find the most error each fold
    local accuracy = 0

    for _,tab in ipairs(validation_list[j]) do
      result_tab[count_fold] = {}
      for i=1,#tab.validation do
        result_tab[count_fold][i] = {}
        for j=1,#tab.test do
          result_tab[count_fold][i][j] = {
            index = tab.validation[i].index,
            from = tab.test[j].index,
            range = euclidean(tab.validation[i], tab.test[j]), -- try another range
            label = tab.test[j].y,
            true_label = tab.validation[i].y
          }
        end
      end
      count_fold = count_fold + 1
    end

    for i=1,k_fold do -- k-fold cross validation
      for j=1, #result_tab[#result_tab] do -- data validation
        insertion_sort(result_tab[i][j])
        local knearest_data = get_k_neighbor(result_tab[i][j], knn)
        is_label_similar = (
            generate_new_label(knearest_data) == result_tab[i][j][count_cross].true_label
        )

        if not is_label_similar then
          count_label = count_label + 1
        end

        is_label_similar = true
        count_cross = count_cross + 1
      end
      best_folds[i] = count_label -- do save here, most_error
      --print("fold " .. i,"error check",count_label, "---------------------------")
      count_label = 0
      count_cross = 1
    end

    local error_label = 1
    local most_error = best_folds[error_label]
    for i=2,#best_folds do
      if best_folds[i] > most_error then
        most_error = best_folds[i]
        error_label = i
      end
    end
    --print("random data: " .. j, "k-fold: " .. k_fold, "\nknn: " .. knn)
    --print("fold: " .. error_label,"high error: " .. most_error)
    accuracy = 100 - most_error/#result_tab[#result_tab]*100
    average = average + accuracy
    --print("accuracy: " .. accuracy .. " %\n")
    -- do save average or not avg but the less accuracy that will be choosen
    less_accuracy[j] = {
      fold = error_label,
      accuracy = accuracy,
      irdata = j
    }

  end
  -- do save total each k
  k_select[count_k] = {
        k = knn,
        info = best_accuracy(less_accuracy),--average / do_iterate -- base on top iterate
        average = average / #validation_list
  }

  print("iterate: " .. count_k, "in random data: "..k_select[count_k].info.irdata,
      "k: "..k_select[count_k].k, k_select[count_k].info.accuracy, "Final avg:" .. 
      k_select[count_k].average,"\n")

  average = 0
  count_k = count_k + 1
  range = range + 1
end

print("Execute to test_list")
knn = {}
knn = generate_k_from_validation(k_select)
print("picked k from validation", knn.k, "that accuracy " .. knn.average)
result_tab = {}
local answer_list = {}
local counter = 0
for _,v_vl in ipairs(test_list) do
  for _,v_tl in ipairs(train_list) do
    table.insert(result_tab, {
      index = v_vl.index,
      range = euclidean(v_vl, v_tl),
      label = v_tl.y
    })
    counter = counter + 1
  end
  -- Do sorting in result_tab to take the nearest/min range
  insertion_sort(result_tab)
  local knn_data = get_k_neighbor(result_tab, knn.k)
  table.insert(answer_list, generate_new_label(knn_data))

  result_tab = {} -- reset
end
table_to_CSV(ANSWER_PATH, answer_list) -- save to CSV file
