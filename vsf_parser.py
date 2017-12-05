import vcf, csv

def flatten(x):
    if type(x) == type([]):
        x = ','.join(map(str, x))
    return x

def readVCF(vcf_input, csv_output, user_id, header=True):
  reader = vcf.Reader(open(vcf_input, 'r'))
  formats = reader.formats.keys()
  infos = reader.infos.keys()
  header = ["SAMPLE"] + list(formats) + ['FILTER', 'CHROM', 'POS', 'REF', 'ALT', 'ID'] + ['info.' + x for x in infos]
  out = csv.writer(open(csv_output, 'w'), delimiter='\t')
  if header:
    out.writerow(header)
  for record in reader:
      info_row = [flatten(record.INFO.get(x, None)) for x in infos]
      fixed = [record.CHROM, record.POS, record.REF, record.ALT, record.ID]
      for sample in record.samples:
          row = [sample.sample]
          row += [flatten(getattr(sample.data, x, None)) for x in formats]
          row += [record.FILTER or '.']
          row += fixed
          row += info_row
          if (row[4] != '0/0') and (row[0] == user_id):
            out.writerow(row)

if __name__ == '__main__':
  normal_blood_vcf = "data/vcfs/G91716.vcf"
  sibling_tumor_vcf = "data/vcfs/G97552.vcf"
  output_tumor = "data/csvs/tumor.csv"
  output_normal = "data/csvs/normal.csv"
  output_brother = "data/csvs/brother.csv"
  tumor_id = 'OF_010116NF2_a'
  brother_id = 'NF2_XY_s'
  normal_id = 'OF_112015SJIA_2'
  flatten_vcf(normal_blood_vcf, output_normal, normal_id)