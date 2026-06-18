insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values
  (
    'model-artifacts',
    'model-artifacts',
    false,
    26214400,
    array['application/json', 'application/octet-stream', 'text/plain']::text[]
  ),
  (
    'evaluation-reports',
    'evaluation-reports',
    false,
    10485760,
    array['application/json', 'text/plain', 'text/markdown']::text[]
  ),
  (
    'public-demo-assets',
    'public-demo-assets',
    true,
    5242880,
    array['image/png', 'image/jpeg', 'text/plain', 'application/json']::text[]
  )
on conflict (id) do update
set
  public = excluded.public,
  file_size_limit = excluded.file_size_limit,
  allowed_mime_types = excluded.allowed_mime_types;
